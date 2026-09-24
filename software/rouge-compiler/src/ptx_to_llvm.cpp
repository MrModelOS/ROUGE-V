// ROUGE-V ptx2ir: PTX -> textual LLVM IR translation.
//
// Stage 1 of the ROUGE-V AOT compiler pipeline (see docs/06-compiler-architecture.md):
// a CUDA kernel's PTX bytecode is parsed by the shared rouge-ptx front-end and
// re-emitted as LLVM IR. The IR is then compiled to native code (clang/LLVM
// backend) instead of being executed by an instruction interpreter.
//
// Design notes:
//  * Every PTX virtual register becomes an `alloca` in the entry block. This is
//    deliberately simple and always correct (memory semantics survive control
//    flow and register redefinition); LLVM's reg2mem/mem2reg promos can later
//    promote them, and the MLIR middle-end (Stage 2) will replace the whole
//    scalar lowering with a SIMT -> RVV vector lowering.
//  * f32 values live in i32 allocas (PTX keeps floats in .b32 registers); float
//    arithmetic is bitcast through `float` in IR.
//  * 32/64-bit integer and float arithmetic, compares, global/shared loads and
//    stores, param loads, branches, predicates, .shared scratchpad addressing,
//    bar.sync -> runtime barrier, atom.add/red.add -> atomicrmw, and f16/bf16
//    arithmetic + conversions are supported. Warp shuffles, CAS/exch atomics,
//    16-bit vector (.f16x2) forms and anything outside the listed subset are
//    rejected with a clear error (TODO list in the compiler README) rather
//    than silently miscompiling.

#include <cctype>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "rougecomp/ptx_to_llvm.h"

namespace rougecomp {
namespace {

// Byte offsets of the CUDA special registers inside the launch descriptor.
const char* kSpecNames[12] = {
    "tid.x",   "tid.y",   "tid.z",   "ctaid.x", "ctaid.y", "ctaid.z",
    "ntid.x",  "ntid.y",  "ntid.z",  "nctaid.x", "nctaid.y", "nctaid.z",
};

int spec_offset(const std::string& name) {
  for (int i = 0; i < 12; ++i)
    if (kSpecNames[i] == name) return 4 * i;
  return -1;
}

bool is_register(const std::string& tok) {
  if (tok.empty() || !std::isalpha(static_cast<unsigned char>(tok[0]))) return false;
  for (char c : tok)
    if (!std::isalnum(static_cast<unsigned char>(c))) return false;
  // Must end with at least one digit to look like %r0/%rd1/%p2.
  return std::isdigit(static_cast<unsigned char>(tok.back()));
}

bool is_numeric(const std::string& tok) {
  if (tok.empty()) return false;
  size_t i = 0;
  if (tok[0] == '-') i = 1;
  return i < tok.size() && std::isdigit(static_cast<unsigned char>(tok[i]));
}

std::string strip_brackets(const std::string& tok) {
  if (tok.size() >= 2 && tok.front() == '[' && tok.back() == ']')
    return tok.substr(1, tok.size() - 2);
  return tok;
}

// ---------------------------------------------------------------------------

class LlvmGen {
 public:
  LlvmGen(const rouge::PtxFunction* fn, std::string* error)
      : fn_(fn), error_(error) {}

  std::string generate() {
    if (fn_->body.empty()) {
      fail("kernel has an empty body");
      return "";
    }
    build_block_names();
    emit_header();
    emit_signature();
    emit_allocas();
    for (size_t i = 0; i < fn_->body.size(); ++i) {
      const auto& ins = fn_->body[i];
      if (i > 0 && block_name_[i] != block_name_[i - 1]) emit_label(block_name_[i]);
      if (!emit_instruction(ins, i)) return "";
      emit_terminator(ins, i);
    }
    if (need_ret_block_) {
      emit_label("b_ret");
      line("  ret void");
    }
    line("}");
    emit_shared_query();
    return out_;
  }

 private:
  const rouge::PtxFunction* fn_;
  std::string* error_;
  std::string out_;
  int uid_ = 0;
  std::vector<std::string> block_name_;  // LLVM block name per PTX instruction
  int anon_seq_ = 0;
  bool need_ret_block_ = false;
  bool use_sync_ = false;  // kernel contains bar.sync -> declare runtime helper

  // Record the error message; returns false so bool helpers can do
  // `return fail("...")`.
  bool fail(const std::string& msg) {
    if (error_) *error_ = msg;
    return false;
  }

  void line(const std::string& s = "") { out_ += s + "\n"; }
  std::string fresh(const char* base) {
    return std::string(base) + std::to_string(uid_++);
  }

  // ---- register typing ---------------------------------------------------

  int reg_bits(const std::string& reg) const {
    const auto it = fn_->regBits.find(reg);
    return it == fn_->regBits.end() ? 32 : it->second;
  }
  std::string reg_type(const std::string& reg) const {
    const int bits = reg_bits(reg);
    if (bits == 1) return "i1";
    return bits == 64 ? "i64" : bits == 16 ? "i16" : "i32";
  }
  static std::string align_of(const std::string& ty) {
    return ty == "i1" ? "1" : ty == "i16" ? "2" : ty == "i64" ? "8" : "4";
  }

  // ---- value access ------------------------------------------------------

  // Load a register's alloca content as an SSA value.
  std::string load_reg(const std::string& reg) {
    const std::string ty = reg_type(reg);
    const std::string v = fresh("%v");
    line("  " + v + " = load " + ty + ", ptr %" + reg + ", align " + align_of(ty));
    return v;
  }
  void store_reg(const std::string& reg, const std::string& ssa) {
    const std::string ty = reg_type(reg);
    line("  store " + ty + " " + ssa + ", ptr %" + reg + ", align " + align_of(ty));
  }
  // Load a register and reinterpret its i32 bits as float.
  std::string load_f32(const std::string& reg) {
    const std::string bits = load_reg(reg);
    const std::string f = fresh("%f");
    line("  " + f + " = bitcast i32 " + bits + " to float");
    return f;
  }
  void store_f32(const std::string& reg, const std::string& fssa) {
    const std::string bits = fresh("%b");
    line("  " + bits + " = bitcast float " + fssa + " to i32");
    store_reg(reg, bits);
  }
  // f16/bf16 live in i16 allocas (PTX .b16 registers) and are bitcast to the
  // LLVM half / bfloat types for arithmetic.
  std::string load16(const std::string& reg, const char* irTy) {
    const std::string bits = load_reg(reg);
    const std::string v = fresh("%h");
    line("  " + v + " = bitcast i16 " + bits + " to " + irTy);
    return v;
  }
  void store16(const std::string& reg, const std::string& v, const char* irTy) {
    const std::string bits = fresh("%b");
    line("  " + bits + " = bitcast " + std::string(irTy) + " " + v + " to i16");
    store_reg(reg, bits);
  }
  // Operand that is either a register or a numeric immediate; returns the value
  // token ("%v3" or "5"); the caller supplies the LLVM type.
  std::string operand(const std::string& tok) {
    if (is_numeric(tok)) return tok;
    return load_reg(tok);
  }

  // CUDA special register read from the launch descriptor (always i32).
  std::string load_spec(const std::string& name) {
    const int off = spec_offset(name);
    const std::string gep = fresh("%lg");
    line("  " + gep + " = getelementptr inbounds i8, ptr %launch, i64 " +
         std::to_string(off));
    const std::string v = fresh("%v");
    line("  " + v + " = load i32, ptr " + gep + ", align 4");
    return v;
  }

  // ---- blocks ------------------------------------------------------------

  bool is_terminator(const rouge::PtxInstruction& ins) const {
    return ins.op == "bra" || ins.op == "bra.uni" || ins.op == "ret" ||
           ins.op == "exit";
  }

  void build_block_names() {
    const size_t n = fn_->body.size();
    block_name_.assign(n, "");
    bool term_prev = true;
    for (size_t i = 0; i < n; ++i) {
      const std::string lab = label_at(i);
      if (i == 0 && !lab.empty())
        block_name_[0] = "b_" + lab;  // entry falls through to this target
      else if (i == 0)
        block_name_[0] = "entry";
      else if (!lab.empty())
        block_name_[i] = "b_" + lab;
      else if (term_prev)
        block_name_[i] = "ba" + std::to_string(anon_seq_++);
      else
        block_name_[i] = block_name_[i - 1];
      term_prev = is_terminator(fn_->body[i]);
    }
  }

  // Label string attached to body index i, or "".
  std::string label_at(size_t i) const {
    for (const auto& kv : fn_->labels)
      if (static_cast<size_t>(kv.second) == i) return kv.first;
    return "";
  }

  void emit_label(const std::string& name) { line(name + ":"); }

  // ---- shared-memory metadata export -------------------------------------
  // The .shared layout lives in the IR, not in the driver: every translated
  // kernel exports
  //   i32 @__rouge_<kernel>_query(ptr %info) -> RougeKernelInfo
  // so the runtime (or a test driver) learns the scratchpad size and symbol
  // offsets straight from the PTX. See runtime/rouge_runtime.h for the ABI.

  std::string meta_prefix(const std::string& suffix) const {
    return "__rouge_" + fn_->name + "_" + suffix;
  }

  void emit_shared_metadata() {
    line("; ---- kernel metadata: .shared layout (rouge kernel info) ----");
    for (size_t i = 0; i < fn_->sharedVars.size(); ++i) {
      const auto& sv = fn_->sharedVars[i];
      line("@.rouge.sym." + fn_->name + "." + sv.name +
           " = private unnamed_addr constant [" + std::to_string(sv.name.size() + 1) +
           " x i8] c\"" + sv.name + "\\00\", align 1");
      (void)i;
    }
    const std::string ty = "{ ptr, i32, i32, i32, i32, i32 }";
    if (fn_->sharedVars.empty()) {
      line("@" + meta_prefix("shared_vars") + " = internal constant [0 x " + ty +
           "] zeroinitializer");
      return;
    }
    std::string body;
    for (size_t i = 0; i < fn_->sharedVars.size(); ++i) {
      const auto& sv = fn_->sharedVars[i];
      if (i) body += ", ";
      body += "{ ptr, i32, i32, i32, i32, i32 } { ptr @.rouge.sym." + fn_->name +
              "." + sv.name + ", i32 " + std::to_string(sv.offset) + ", i32 " +
              std::to_string(sv.elemBytes * sv.count) + ", i32 " +
              std::to_string(sv.align) + ", i32 " + std::to_string(sv.elemBytes) +
              ", i32 " + std::to_string(sv.count) + " }";
    }
    line("@" + meta_prefix("shared_vars") + " = internal constant [" +
         std::to_string(fn_->sharedVars.size()) + " x " + ty + "] [" + body + "]");
  }

  void emit_shared_query() {
    // define i32 @__rouge_<kernel>_query(ptr %info)
    line("define i32 @" + meta_prefix("query") + "(ptr %info) {");
    line("entry:");
    const std::string p0 = fresh("%fi0");
    line("  " + p0 + " = getelementptr inbounds i8, ptr %info, i64 0");
    line("  store i32 " + std::to_string(fn_->sharedSize) + ", ptr " + p0 +
         ", align 4");
    const std::string p1 = fresh("%fi1");
    line("  " + p1 + " = getelementptr inbounds i8, ptr %info, i64 4");
    line("  store i32 " + std::to_string(fn_->sharedVars.size()) + ", ptr " + p1 +
         ", align 4");
    const std::string p2 = fresh("%fi2");
    line("  " + p2 + " = getelementptr inbounds i8, ptr %info, i64 8");
    line("  store ptr @" + meta_prefix("shared_vars") + ", ptr " + p2 + ", align 8");
    line("  ret i32 1");
    line("}");
  }

  // ---- header / allocas --------------------------------------------------

  void emit_header() {
    line("; Generated by ROUGE-V ptx2ir — AOT path, do not edit");
    line("; PTX kernel: " + fn_->name);
    line("source_filename = \"" + fn_->name + ".ptx\"");
    line("target triple = \"x86_64-pc-linux-gnu\"");
    emit_shared_metadata();
  }

  void emit_signature() {
    // Module-level declare for the block barrier runtime helper (bar.sync).
    for (const auto& ins : fn_->body)
      if (ins.op == "bar.sync" || ins.op.rfind("bar.", 0) == 0) use_sync_ = true;
    if (use_sync_) line("declare void @__rouge_syncthreads(ptr)");
    std::string sig = "define void @" + fn_->name + "(";
    for (size_t i = 0; i < fn_->params.size(); ++i) {
      if (i) sig += ", ";
      sig += param_llvm_type(fn_->params[i]) + " %arg" + std::to_string(i);
    }
    if (!fn_->params.empty()) sig += ", ";
    sig += "ptr %launch) {";
    line(sig);
  }

  static std::string param_llvm_type(const rouge::PtxParam& p) {
    // PTX params arrive as u64/s64/b64/u32/s32/b32/f32; the parser stores the
    // width, and we can no longer distinguish f32 from b32 here.
    return p.width == 8 ? "i64" : "i32";
  }

  void emit_allocas() {
    std::set<std::string> used;
    const auto is_label = [this](const std::string& tok) {
      return fn_->labels.find(tok) != fn_->labels.end();
    };
    const auto add_reg = [&used, &is_label](const std::string& tok) {
      if (is_register(tok) && !is_label(tok)) used.insert(tok);
    };
    for (const auto& ins : fn_->body) {
      if (!ins.pred.empty()) add_reg(ins.pred[0] == '!' ? ins.pred.substr(1)
                                                        : ins.pred);
      for (const auto& a : ins.args) {
        const std::string inner = strip_brackets(a);
        if (inner == a)
          add_reg(a);
        else
          add_reg(inner);  // memory operand "[%rd1]"
      }
    }
    // Entry block label.
    if (block_name_.empty() || block_name_[0] != "entry") {
      line("entry:");
    }
    for (const auto& r : used) {
      const std::string ty = reg_type(r);
      line("  %" + r + " = alloca " + ty + ", align " + align_of(ty));
    }
    if (!block_name_.empty() && block_name_[0] != "entry") {
      // First instruction is a branch target: jump to its block.
      line("  br label %" + block_name_[0]);
    }
  }

  // ---- instruction lowering ---------------------------------------------

  bool emit_instruction(const rouge::PtxInstruction& ins, size_t idx) {
    const std::string op = ins.op;
    const auto& A = ins.args;

    if (op == "ld.param.u64" || op == "ld.param.s64" || op == "ld.param.b64") {
      return emit_ld_param(ins, /*bits=*/64);
    }
    if (op == "ld.param.u32" || op == "ld.param.s32" || op == "ld.param.b32") {
      return emit_ld_param(ins, /*bits=*/32);
    }
    if (op == "ld.param.f32") {
      return emit_ld_param(ins, /*bits=*/32);  // f32 params are stored as bits
    }

    if (op == "mov.u64" || op == "mov.s64" || op == "mov.b64" ||
        op == "mov.u32" || op == "mov.s32" || op == "mov.b32" ||
        op == "mov.u16" || op == "mov.s16" || op == "mov.b16" ||
        op == "mov.f16" || op == "mov.bf16") {
      if (A.size() != 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      if (spec_offset(A[1]) >= 0) {
        store_reg(A[0], load_spec(A[1]));
        return true;
      }
      // Width mismatch (e.g. mov.b16 %h, %r32) -> truncate the bit pattern.
      const std::string dstTy = reg_type(A[0]);
      std::string val = operand(A[1]);  // register or numeric immediate
      if (is_register(A[1]) && reg_type(A[1]) != dstTy) {
        const std::string t = fresh("%t");
        line("  " + t + " = trunc " + reg_type(A[1]) + " " + val + " to " + dstTy);
        val = t;
      }
      store_reg(A[0], val);
      return true;
    }
    if (op == "mov.f32") {
      if (A.size() != 2 || !is_register(A[1]))
        return fail("bad " + op + " operands @ " + std::to_string(idx));
      store_reg(A[0], load_reg(A[1]));  // bit pattern copy
      return true;
    }
    if (op.rfind("cvta.to.shared.", 0) == 0) {
      // cvta.to.shared.u64 %rd, sym  ->  scratchpadBase + varOffset
      if (A.size() != 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      int off = 0, c = 0;
      if (!shared_symbol(A[1], &off, &c))
        return fail("cvta.to.shared of non-shared symbol '" + A[1] + "' @" +
                    std::to_string(idx));
      const std::string base = load_smem_base();
      const std::string addr = fresh("%sa");
      line("  " + addr + " = add i64 " + base + ", " + std::to_string(off + c));
      store_reg(A[0], addr);
      return true;
    }
    if (op == "cvta.to.global.u64" || op == "cvta.to.global.u32" ||
        op == "cvta.to.global.b64" || op == "cvta.to.global.b32") {
      if (A.size() != 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      store_reg(A[0], load_reg(A[1]));
      return true;
    }

    if (op == "mul.wide.u32" || op == "mul.wide.s32") {
      if (A.size() != 3) return fail("bad mul.wide operands @ " + std::to_string(idx));
      const std::string x = fresh("%x");
      const std::string y = fresh("%y");
      const bool is_signed = op.back() == 's';
      const char* zc = is_signed ? "sext" : "zext";
      line("  " + x + " = " + zc + " i32 " + operand(A[1]) + " to i64");
      line("  " + y + " = " + zc + " i32 " + operand(A[2]) + " to i64");
      const std::string m = fresh("%m");
      line("  " + m + " = mul i64 " + x + ", " + y);
      store_reg(A[0], m);
      return true;
    }

    if (op.rfind("add.", 0) == 0 || op.rfind("sub.", 0) == 0 ||
        op.rfind("mul.", 0) == 0 || op.rfind("div.", 0) == 0 ||
        op.rfind("fma.", 0) == 0) {
      return emit_arith(ins, idx);
    }

    if (op.rfind("atom.", 0) == 0 || op.rfind("red.", 0) == 0) {
      return emit_atomic(ins, idx);
    }

    if (op.rfind("cvt.", 0) == 0) {
      return emit_cvt(ins, idx);
    }

    if (op.rfind("setp.", 0) == 0) {
      return emit_setp(ins, idx);
    }

    if (op.rfind("ld.global", 0) == 0 || op.rfind("ld.shared", 0) == 0) {
      return emit_ld_mem(ins, idx);
    }
    if (op.rfind("st.global", 0) == 0 || op.rfind("st.shared", 0) == 0) {
      return emit_st_mem(ins, idx);
    }

    if (op == "bar.sync" || op.rfind("bar.", 0) == 0) {
      // Block-wide barrier: host -> __rouge_syncthreads (pthread_barrier);
      // RISC-V target -> fence / custom inter-core barrier.
      if (A.size() > 0 && is_numeric(A[0])) {
        // bar.sync 0[, count] — ignore barrier id for now (single barrier).
      }
      line("  call void @__rouge_syncthreads(ptr %launch)");
      return true;
    }

    if (op == "bra" || op == "bra.uni" || op == "ret" || op == "exit") return true;

    return fail("unsupported PTX op '" + op + "' @ " + std::to_string(idx) +
                " (vectorizing lowering landing in MLIR stage 2)");
  }

  // =================== helpers for specific ops ===========================

  bool emit_ld_param(const rouge::PtxInstruction& ins, int bits) {
    const auto& A = ins.args;
    if (A.size() != 2) return fail("bad ld.param operands");
    const std::string pname = strip_brackets(A[1]);
    int pidx = -1;
    for (size_t i = 0; i < fn_->params.size(); ++i)
      if (fn_->params[i].name == pname) {
        pidx = static_cast<int>(i);
        break;
      }
    if (pidx < 0) return fail("unknown kernel param '" + pname + "'");
    const std::string src = "%arg" + std::to_string(pidx);
    const std::string dstty = reg_type(A[0]);
    const std::string srcty = param_llvm_type(fn_->params[pidx]);
    if (dstty == srcty) {
      store_reg(A[0], src);
    } else {
      const std::string tmp = fresh("%t");
      line("  " + tmp + " = bitcast " + srcty + " " + src + " to " + dstty);
      store_reg(A[0], tmp);
    }
    (void)bits;
    return true;
  }

  bool emit_arith(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    const bool fma = op.rfind("fma.", 0) == 0;
    if (A.size() != (fma ? 4u : 3u))
      return fail("bad arithmetic operands @ " + std::to_string(idx));

    // f16 / bf16: widen to f32, compute, narrow once (fptrunc, RNE). The
    // interpreter does the same, so both paths agree bit-for-bit; on RISC-V
    // the target's Zfh/Zfbf instructions take over when available.
    const bool f16 = op.find(".f16") != std::string::npos;
    const bool bf16 = op.find(".bf16") != std::string::npos;
    if (f16 || bf16) {
      const char* irTy = f16 ? "half" : "bfloat";
      for (const auto& a : A)
        if (is_register(a) && reg_bits(a) != 16)
          return fail(op + " needs 16-bit registers (@ " + std::to_string(idx) + ")");
      const auto widen = [&](const std::string& r) {
        const std::string h = load16(r, irTy);
        const std::string f = fresh("%f");
        line("  " + f + " = fpext " + std::string(irTy) + " " + h + " to float");
        return f;
      };
      const std::string x = widen(A[1]);
      const std::string y = widen(A[2]);
      const char* ll = fma
                           ? nullptr  // llvm.fma intrinsic call (see below)
                           : (op[0] == 'a' ? "fadd"
                                           : op[0] == 's' ? "fsub"
                                                           : op[0] == 'm' ? "fmul" : "fdiv");
      const std::string r = fresh("%r");
      if (fma) {
        const std::string z = widen(A[3]);
        line("  " + r + " = call float @llvm.fma.f32(float " + x + ", float " + y +
             ", float " + z + ")");
      } else {
        line("  " + r + " = " + ll + " float " + x + ", " + y);
      }
      const std::string h = fresh("%h");
      line("  " + h + " = fptrunc float " + r + " to " + irTy);
      store16(A[0], h, irTy);
      return true;
    }

    const bool f32 = op.rfind(".f32") != std::string::npos;
    if (f32) {
      const std::string x = load_f32(A[1]);
      const std::string y = load_f32(A[2]);
      const std::string r = fresh("%r");
      const char* ll = op[0] == 'a' ? "fadd" : op[0] == 's' ? "fsub"
                                     : op[0] == 'm' ? "fmul"
                                                    : "fdiv";
      line("  " + r + " = " + ll + " float " + x + ", " + y);
      store_f32(A[0], r);
      return true;
    }
    // Integer: u32/s32/u64/s64.
    const std::string ty = reg_type(A[0]);
    const char* ll = op[0] == 'a' ? "add" : op[0] == 's' ? "sub" : "mul";
    const std::string x = operand(A[1]);
    const std::string y = operand(A[2]);
    const std::string r = fresh("%r");
    line("  " + r + " = " + ll + " " + ty + " " + x + ", " + y);
    store_reg(A[0], r);
    (void)idx;
    return true;
  }

  bool emit_atomic(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    if (op.find(".add.") == std::string::npos)
      return fail("unsupported atomic '" + op + "' @ " + std::to_string(idx) +
                  " (supported: atom.add.{u32,s32,u64,s64,f32}, red.add.*)");
    const bool has_dst = op.rfind("atom.", 0) == 0;
    const size_t ai = has_dst ? 1 : 0;
    if (A.size() < ai + 2)
      return fail("bad atomic operands @ " + std::to_string(idx));
    const std::string ty = op.substr(op.rfind('.') + 1);
    const std::string addr = resolve_addr(A[ai]);
    const bool w64 =
        ty == "u64" || ty == "s64" || ty == "b64";
    const int align = w64 ? 8 : 4;
    const char* llTy = w64 ? "i64" : "i32";
    // atom.add is relaxed in PTX -> LLVM "monotonic" ordering.
    const std::string old = fresh("%old");
    if (ty == "f32") {
      const std::string f = fresh("%f");
      line("  " + f + " = bitcast i32 " + operand(A[ai + 1]) + " to float");
      line("  " + old + " = atomicrmw fadd ptr " + addr + ", float " + f +
           " monotonic, align 4");
    } else if (ty == "u32" || ty == "s32" || ty == "b32" || w64) {
      line("  " + old + " = atomicrmw add ptr " + addr + ", " + llTy + " " +
           operand(A[ai + 1]) + " monotonic, align " + std::to_string(align));
    } else {
      return fail("unsupported atomic type '" + ty + "' @ " + std::to_string(idx));
    }
    if (has_dst) {
      if (ty == "f32") {
        const std::string bits = fresh("%b");
        line("  " + bits + " = bitcast float " + old + " to i32");
        store_reg(A[0], bits);
      } else {
        store_reg(A[0], old);
      }
    }
    return true;
  }

  bool emit_cvt(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    if (A.size() != 2) return fail("bad cvt operands @ " + std::to_string(idx));
    std::string spec = ins.op.substr(4);  // "u64.u32", "u32.u64", "rn.bf16.f32"...
    if (spec.rfind("rn.", 0) == 0) spec = spec.substr(3);
    const std::string dst = spec.substr(0, spec.find('.'));
    const std::string src = spec.substr(spec.find('.') + 1);

    // f16 / bf16 conversions: LLVM half / bfloat with fpext / fptrunc, which
    // round exactly like the interpreter's software conversions.
    const bool src16 = src == "f16" || src == "bf16";
    const bool dst16 = dst == "f16" || dst == "bf16";
    if (dst16 && is_register(A[0]) && reg_bits(A[0]) != 16)
      return fail(ins.op + " needs a 16-bit destination (@ " +
                  std::to_string(idx) + ")");
    if (src16 && is_register(A[1]) && reg_bits(A[1]) != 16)
      return fail(ins.op + " needs a 16-bit source (@ " + std::to_string(idx) + ")");
    if (dst16) {
      const char* irTy = dst == "f16" ? "half" : "bfloat";
      if (src == "f32" || src == "f64") {
        if (is_register(A[1]) && reg_bits(A[1]) != 32)
          return fail(ins.op + " needs a 32-bit float source (@ " +
                      std::to_string(idx) + ")");
        const std::string f = load_f32(A[1]);
        const std::string h = fresh("%h");
        line("  " + h + " = fptrunc float " + f + " to " + irTy);
        store16(A[0], h, irTy);
        return true;
      }
      if (src == "u32" || src == "s32" || src == "u16" || src == "s16") {
        const std::string v = operand(A[1]);
        const std::string f = fresh("%f");
        line("  " + f + " = " + (src[0] == 'u' ? "uitofp" : "sitofp") + " " +
             reg_type(A[1]) + " " + v + " to float");
        const std::string h = fresh("%h");
        line("  " + h + " = fptrunc float " + f + " to " + irTy);
        store16(A[0], h, irTy);
        return true;
      }
      if (src == "f16" || src == "bf16") {
        // reinterpret the 16-bit pattern through f32
        const char* srcIr = src == "f16" ? "half" : "bfloat";
        const std::string h = load16(A[1], srcIr);
        const std::string f = fresh("%f");
        line("  " + f + " = fpext " + std::string(srcIr) + " " + h + " to float");
        const std::string out = fresh("%h");
        line("  " + out + " = fptrunc float " + f + " to " + irTy);
        store16(A[0], out, irTy);
        return true;
      }
    }
    if (src16 && dst == "f32") {
      const char* srcIr = src == "f16" ? "half" : "bfloat";
      const std::string h = load16(A[1], srcIr);
      const std::string f = fresh("%f");
      line("  " + f + " = fpext " + std::string(srcIr) + " " + h + " to float");
      store_f32(A[0], f);
      return true;
    }
    if (src16 && (dst == "u32" || dst == "s32" || dst == "u16" || dst == "s16")) {
      const char* srcIr = src == "f16" ? "half" : "bfloat";
      const std::string h = load16(A[1], srcIr);
      const std::string f = fresh("%f");
      line("  " + f + " = fpext " + std::string(srcIr) + " " + h + " to float");
      const std::string i = fresh("%i");
      line("  " + i + " = " + (dst[0] == 'u' ? "fptoui" : "fptosi") +
           " float " + f + " to " + reg_type(A[0]));
      store_reg(A[0], i);
      return true;
    }
    const auto pair = [](const std::string& a, const std::string& b) {
      return a + "." + b;
    };
    if (pair(dst, src) == "u64.u32" || pair(dst, src) == "s64.u32") {
      const std::string z = fresh("%z");
      line("  " + z + " = zext i32 " + operand(A[1]) + " to i64");
      store_reg(A[0], z);
      return true;
    }
    if (pair(dst, src) == "s64.s32") {
      const std::string z = fresh("%z");
      line("  " + z + " = sext i32 " + operand(A[1]) + " to i64");
      store_reg(A[0], z);
      return true;
    }
    if (pair(dst, src) == "u32.u64" || pair(dst, src) == "s32.u64") {
      const std::string t = fresh("%t");
      line("  " + t + " = trunc i64 " + operand(A[1]) + " to i32");
      store_reg(A[0], t);
      return true;
    }
    if (pair(dst, src) == "f32.u32") {
      const std::string f = fresh("%f");
      line("  " + f + " = uitofp i32 " + operand(A[1]) + " to float");
      store_f32(A[0], f);
      return true;
    }
    if (pair(dst, src) == "f32.s32") {
      const std::string f = fresh("%f");
      line("  " + f + " = sitofp i32 " + operand(A[1]) + " to float");
      store_f32(A[0], f);
      return true;
    }
    if (pair(dst, src) == "u32.f32") {
      const std::string i = fresh("%i");
      line("  " + i + " = fptoui float " + load_f32(A[1]) + " to i32");
      store_reg(A[0], i);
      return true;
    }
    if (pair(dst, src) == "s32.f32") {
      const std::string i = fresh("%i");
      line("  " + i + " = fptosi float " + load_f32(A[1]) + " to i32");
      store_reg(A[0], i);
      return true;
    }
    return fail("unsupported cvt '" + spec + "' @ " + std::to_string(idx));
  }

  bool emit_setp(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    if (A.size() != 3) return fail("bad setp operands @ " + std::to_string(idx));
    std::string spec = ins.op.substr(5);  // e.g. "ge.u32"
    const std::string cond = spec.substr(0, spec.find('.'));
    const std::string ll =
        cond == "ge" ? "uge" : cond == "gt" ? "ugt" : cond == "lt" ? "ult"
                             : cond == "le" ? "ule" : cond == "eq" ? "eq"
                                                                   : "ne";
    const std::string x = operand(A[1]);
    const std::string y = operand(A[2]);
    const std::string c = fresh("%c");
    line("  " + c + " = icmp " + ll + " i32 " + x + ", " + y);
    store_reg(A[0], c);
    return true;
  }

  std::string addr_of(const std::string& reg_tok) {
    const std::string a = load_reg(reg_tok);
    const std::string p = fresh("%p");
    line("  " + p + " = inttoptr i64 " + a + " to ptr");
    return p;
  }

  // Read the block scratchpad base (launch descriptor, offset 48) as i64.
  std::string load_smem_base() {
    const std::string gep = fresh("%lg");
    line("  " + gep + " = getelementptr inbounds i8, ptr %launch, i64 48");
    const std::string v = fresh("%v");
    line("  " + v + " = load i64, ptr " + gep + ", align 8");
    return v;
  }

  // If the operand names a .shared var (optionally with a +/-Const byte
  // suffix), return its offset and the constant part.
  bool shared_symbol(const std::string& inner, int* varOffset, int* constOff) const {
    for (const auto& sv : fn_->sharedVars) {
      if (inner == sv.name) {
        *varOffset = sv.offset;
        *constOff = 0;
        return true;
      }
      if (inner.size() > sv.name.size() &&
          inner.compare(0, sv.name.size(), sv.name) == 0) {
        const char c = inner[sv.name.size()];
        if (c == '+' || c == '-') {
          *varOffset = sv.offset;
          *constOff = std::atoi(inner.c_str() + sv.name.size());
          return true;
        }
      }
    }
    return false;
  }

  // Resolve a memory operand ("[%rd1]", "[smem]", "[smem+4]") to a `ptr` SSA:
  // shared symbols are scratchpad-relative, registers hold addresses as-is.
  std::string resolve_addr(const std::string& tok) {
    const std::string inner = strip_brackets(tok);
    int varOffset = 0, constOff = 0;
    if (shared_symbol(inner, &varOffset, &constOff)) {
      const std::string base = load_smem_base();
      const std::string off = fresh("%a");
      line("  " + off + " = add i64 " + base + ", " +
           std::to_string(varOffset + constOff));
      const std::string p = fresh("%p");
      line("  " + p + " = inttoptr i64 " + off + " to ptr");
      return p;
    }
    return addr_of(inner);
  }

  bool is_16bit_ty(const std::string& op) const {
    return op.find(".f16") != std::string::npos ||
           op.find(".b16") != std::string::npos ||
           op.find(".u16") != std::string::npos ||
           op.find(".s16") != std::string::npos ||
           op.find(".bf16") != std::string::npos;
  }

  bool emit_ld_mem(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    if (A.size() != 2) return fail("bad ld.* operands @ " + std::to_string(idx));
    const bool f32 = ins.op.rfind(".f32") != std::string::npos;
    const bool w64 = ins.op.rfind(".u64") != std::string::npos ||
                     ins.op.rfind(".s64") != std::string::npos ||
                     ins.op.rfind(".b64") != std::string::npos;
    std::string addr = resolve_addr(A[1]);
    if (is_16bit_ty(ins.op)) {
      if (reg_bits(A[0]) != 16)
        return fail(ins.op + " needs a 16-bit destination register @" +
                    std::to_string(idx));
      const std::string v = fresh("%v");
      line("  " + v + " = load i16, ptr " + addr + ", align 2");
      store_reg(A[0], v);
      return true;
    }
    if (f32) {
      const std::string v = fresh("%f");
      line("  " + v + " = load float, ptr " + addr + ", align 4");
      store_f32(A[0], v);
      return true;
    }
    if (w64) {
      const std::string v = fresh("%v");
      line("  " + v + " = load i64, ptr " + addr + ", align 8");
      store_reg(A[0], v);
      return true;
    }
    const std::string v = fresh("%v");
    line("  " + v + " = load i32, ptr " + addr + ", align 4");
    store_reg(A[0], v);
    (void)idx;
    return true;
  }

  bool emit_st_mem(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    if (A.size() != 2) return fail("bad st.* operands @ " + std::to_string(idx));
    const bool f32 = ins.op.rfind(".f32") != std::string::npos;
    const bool w64 = ins.op.rfind(".u64") != std::string::npos ||
                     ins.op.rfind(".s64") != std::string::npos ||
                     ins.op.rfind(".b64") != std::string::npos;
    std::string addr = resolve_addr(A[0]);
    if (is_16bit_ty(ins.op)) {
      if (reg_bits(A[1]) != 16)
        return fail(ins.op + " needs a 16-bit source register @" +
                    std::to_string(idx));
      line("  store i16 " + load_reg(A[1]) + ", ptr " + addr + ", align 2");
      return true;
    }
    if (f32) {
      const std::string bits = load_reg(A[1]);
      const std::string f = fresh("%f");
      line("  " + f + " = bitcast i32 " + bits + " to float");
      line("  store float " + f + ", ptr " + addr + ", align 4");
      return true;
    }
    if (w64) {
      line("  store i64 " + load_reg(A[1]) + ", ptr " + addr + ", align 8");
      return true;
    }
    line("  store i32 " + load_reg(A[1]) + ", ptr " + addr + ", align 4");
    (void)idx;
    return true;
  }

  // ---- terminator --------------------------------------------------------

  void emit_terminator(const rouge::PtxInstruction& ins, size_t i) {
    if (ins.op == "ret" || ins.op == "exit") {
      line("  ret void");
      return;
    }
    if (ins.op == "bra" || ins.op == "bra.uni") {
      if (ins.args.empty()) return;  // invalid; caught upstream
      const std::string target = "b_" + ins.args[0];
      if (!ins.pred.empty()) {
        const std::string pred =
            ins.pred[0] == '!' ? ins.pred.substr(1) : ins.pred;
        std::string pv = load_reg(pred);
        if (ins.pred[0] == '!') {
          const std::string nv = fresh("%n");
          line("  " + nv + " = xor i1 " + pv + ", true");
          pv = nv;
        }
        const std::string fall = (i + 1 < fn_->body.size())
                                     ? block_name_[i + 1]
                                     : ret_block();
        line("  br i1 " + pv + ", label %" + target + ", label %" + fall);
        return;
      }
      line("  br label %" + target);
      return;
    }
    // Non-terminator: fall through. Branch only when a new block begins; when the
    // next instruction stays in the current block, sequential layout suffices.
    if (i + 1 >= fn_->body.size()) {
      line("  ret void");
    } else if (block_name_[i + 1] != block_name_[i]) {
      line("  br label %" + block_name_[i + 1]);
    }
  }

  std::string ret_block() {
    need_ret_block_ = true;
    return "b_ret";
  }
};

}  // namespace

std::string ptx_to_llvm_ir(const rouge::PtxProgram& prog, int fnIndex,
                           std::string* error) {
  if (fnIndex < 0 || static_cast<size_t>(fnIndex) >= prog.functions.size()) {
    if (error) *error = "kernel index out of range";
    return "";
  }
  const rouge::PtxFunction& fn = prog.functions[static_cast<size_t>(fnIndex)];
  LlvmGen gen(&fn, error);
  return gen.generate();
}

}  // namespace rougecomp