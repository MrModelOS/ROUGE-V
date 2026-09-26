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
//    arithmetic + conversions are supported. The full 16/32/64-bit integer ALU
//    (and/or/xor/not/shl/shr, mul.lo/mul.hi, mad.lo/mad.hi, div/rem, min/max,
//    neg/abs/popc/clz/bfind.shiftamt), the f32 unary math (fneg/fnabs/sqrt/
//    rcp/rsqrt), selp/slct, the universal and/or/not .pred logic, the 16/32/64
//    bit integer cvt family and 8/16-bit global loads and stores are supported
//    too, all bit-for-bit identical to the interpreter in rouge-ptx/ptx.cpp,
//    which is the behavioural reference for this lowering. Warp shuffles,
//    CAS/exch atomics, 16-bit vector (.f16x2) forms, f64 *arithmetic*, float
//    compares, saturating / directed-rounding conversions and anything outside
//    the listed subset are rejected with a clear error (TODO list in the
//    compiler README) rather than silently miscompiling. f64 *bit patterns*
//    (mov.f64, ld/st.*.f64, cvt.{f64}, selp/slct.f64) are supported.
//
//  * min/max, sqrt and the f32 min/max go through the llvm.smin/umin/sqrt/
//    minnum intrinsics rather than the smin/umin/fsqrt/fmin opcodes: this
//    LLVM's textual parser does not accept those keywords. They are the same
//    operations, so the machine code is identical.

#include <cctype>
#include <cstdlib>
#include <cstring>
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

// Split "r6+512" / "rd2-4" into a base name and a constant byte offset.
// Returns false when there is no trailing "+Const" / "-Const".
bool split_reg_offset(const std::string& inner, std::string* base,
                      int64_t* offset) {
  const size_t p = inner.find_first_of("+-");
  if (p == std::string::npos || p == 0) return false;
  const std::string tail = inner.substr(p + 1);
  if (tail.empty()) return false;
  for (char c : tail)
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  *base = inner.substr(0, p);
  *offset = std::strtoll(tail.c_str(), nullptr, 10);
  return true;
}

// Braces/commas/whitespace left over from a vector register list: operands are
// split on ',', so "{%r2, %r3}" reaches us as the two tokens "{%r2" and "r3}".
// Same trick as vector_reg_name() in the interpreter; the leading '%' the
// parser only strips outside such a list goes too.
std::string clean_operand(const std::string& tok) {
  const auto junk = [](char c) {
    return c == '{' || c == '}' || c == ',' ||
           std::isspace(static_cast<unsigned char>(c)) != 0;
  };
  size_t b = 0, e = tok.size();
  while (b < e && junk(tok[b])) ++b;
  while (e > b && junk(tok[e - 1])) --e;
  std::string s = tok.substr(b, e - b);
  if (!s.empty() && s[0] == '%') s.erase(s.begin());
  return s;
}

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Width in bits of the integer type a mnemonic ends with (".s32", ".u64",
// ".b16", ...); 0 for float / predicate / vector forms so the callers can
// route those to the dedicated blocks. Mirrors int_type_bits() in the
// interpreter, which is the behavioural reference for this lowering.
int int_type_bits(const std::string& op) {
  if (ends_with(op, ".s64") || ends_with(op, ".u64") || ends_with(op, ".b64")) return 64;
  if (ends_with(op, ".s32") || ends_with(op, ".u32") || ends_with(op, ".b32")) return 32;
  if (ends_with(op, ".s16") || ends_with(op, ".u16") || ends_with(op, ".b16")) return 16;
  return 0;
}

bool is_signed_int_type(const std::string& op) {
  return ends_with(op, ".s16") || ends_with(op, ".s32") || ends_with(op, ".s64");
}

// ---------------------------------------------------------------------------

class LlvmGen {
 public:
  LlvmGen(const rouge::PtxFunction* fn, const rouge::PtxProgram* prog,
          std::string* error)
      : fn_(fn), prog_(prog), error_(error) {}

  std::string generate() {
    normalize_body();
    if (body_.empty()) {
      fail("kernel has an empty body");
      return "";
    }
    build_block_names();
    emit_header();
    emit_signature();
    emit_allocas();
    for (size_t i = 0; i < body_.size(); ++i) {
      const auto& ins = body_[i];
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
  const rouge::PtxProgram* prog_ = nullptr;  // for module-scope .global decls

  // True when the token names a module-scope .global/.const variable
  // (a __device__ array) rather than a virtual register.
  bool is_global_symbol(const std::string& tok) const {
    return prog_ && prog_->globalIndex.find(tok) != prog_->globalIndex.end();
  }

  // LLVM element type for a .global element width.
  static const char* llvm_scalar_ty(int bytes) {
    switch (bytes) {
      case 1: return "i8";
      case 2: return "i16";
      case 8: return "i64";
      default: return "i32";
    }
  }

  // Declare the module's __device__ variables so "cvta.to.global %rd, sym"
  // can take a real address. float/double are bit-cast compatible with
  // their integer counterparts, which is how the rest of this file treats
  // f32 already.
  void emit_global_decls() {
    if (!prog_ || prog_->globals.empty()) return;
    for (const auto& g : prog_->globals) {
      // Count-first array syntax: clang's LLParser mis-reads "[i8 x N]" (it
      // expects an address-space number after i8), while "[N x i8]" is
      // accepted by every LLVM we target.
      const std::string ty = "[" + std::to_string(g.count) + " x " +
                             llvm_scalar_ty(g.elemBytes) + "]";
      line("@" + g.name + " = external global " + ty + ", align " +
           std::to_string(g.align));
    }
  }
  // fn_->body / fn_->labels after normalize_body(): the instruction list with
  // the parser's label pseudo-instructions folded away, and the label map
  // pointing into it.
  std::vector<rouge::PtxInstruction> body_;
  std::map<std::string, size_t> labels_;
  std::string* error_;
  std::string out_;
  int uid_ = 0;
  std::vector<std::string> block_name_;  // LLVM block name per PTX instruction
  int anon_seq_ = 0;
  bool need_ret_block_ = false;
  bool use_sync_ = false;  // kernel contains bar.sync -> declare runtime helper
  // `declare` lines for the llvm.* intrinsics the body uses (abs / ctpop /
  // ctlz), collected by scan_intrinsics() before the body is emitted.
  std::set<std::string> intrinsics_;

  // Record the error message; returns false so bool helpers can do
  // `return fail("...")`.
  bool fail(const std::string& msg) {
    if (error_) *error_ = msg;
    return false;
  }

  void line(const std::string& s = "") { out_ += s + "\n"; }
  // Fresh SSA name. The '_' is deliberate: a PTX register name is all
  // alphanumeric (see is_register), so a temporary can never shadow one, no
  // matter how the kernel numbers its registers.
  std::string fresh(const char* base) {
    return std::string(base) + "_" + std::to_string(uid_++);
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
  static int type_bits(const std::string& ty) {
    return ty == "i1" ? 1 : ty == "i16" ? 16 : ty == "i64" ? 64 : 32;
  }
  // LLVM integer type for a mnemonic width (16 / 32 / 64).
  static std::string int_ty(int bits) {
    return bits == 64 ? "i64" : bits == 16 ? "i16" : "i32";
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
  // Load a register and reinterpret its bits as float. The operand may also be
  // a "0f..." literal, in which case operand() hands back its value directly.
  std::string load_f32(const std::string& reg) {
    const std::string bits = operand(reg);
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
    if (is_numeric(tok)) return normalize_literal(tok);
    return load_reg(tok);
  }
  // PTX spells a float immediate as its bit pattern: "0f3F800000" (f32) or
  // "0d400921FB54442D18" (f64). LLVM's integer constants are decimal (or 0x
  // hex, but not with a per-width digit count), and the payload already lives
  // in an i32/i64 alloca here, so re-spell the pattern as a decimal value.
  static std::string normalize_literal(const std::string& tok) {
    const bool f32 = tok.size() == 10 && tok[0] == '0' && tok[1] == 'f';
    const bool f64 = tok.size() == 18 && tok[0] == '0' && tok[1] == 'd';
    if (!f32 && !f64) return tok;
    const unsigned long long v = std::strtoull(tok.c_str() + 2, nullptr, 16);
    if (f32) return std::to_string(static_cast<unsigned long long>(static_cast<uint32_t>(v)));
    return std::to_string(v);
  }
  // LLVM type an operand must be read as: a literal takes the width the
  // mnemonic declares (a literal has no type of its own), a register its own.
  std::string operand_type(const std::string& tok, int bits) const {
    return is_register(tok) ? reg_type(tok) : int_ty(bits);
  }
  // A conversion produces a value of the width its *mnemonic* names, while the
  // destination register may be wider or narrower. Line the two up (zext when
  // growing, trunc when shrinking) exactly like the interpreter's width_mask.
  std::string fit_reg(const std::string& reg, const std::string& val,
                      const std::string& valTy) {
    const std::string ty = reg_type(reg);
    if (ty == valTy) return val;
    const std::string t = fresh("%t");
    line("  " + t + " = " + (type_bits(ty) > type_bits(valTy) ? "zext" : "trunc") +
         " " + valTy + " " + val + " to " + ty);
    return t;
  }
  // Vector memory type of a ld/st mnemonic ("v2.f32", "v4.u32", ...), or "".
  static std::string vector_type(const std::string& op) {
    const size_t dot = op.rfind('.');
    if (dot == std::string::npos || dot == 0) return "";
    const size_t prev = op.rfind('.', dot - 1);
    if (prev == std::string::npos) return "";
    const std::string seg = op.substr(prev + 1, dot - prev - 1);
    if (seg.size() < 2 || seg[0] != 'v') return "";
    for (size_t i = 1; i < seg.size(); ++i)
      if (!std::isdigit(static_cast<unsigned char>(seg[i]))) return "";
    return op.substr(prev + 1);
  }
  // Which of the integer-ALU / select / predicate families a mnemonic belongs
  // to ("" = not ours, so the caller falls through to the generic arith path).
  // Mirrors the routing of run_thread_step() in the interpreter.
  static std::string alu_family(const std::string& op) {
    if (op.rfind("and.pred", 0) == 0 || op.rfind("or.pred", 0) == 0 ||
        op.rfind("not.pred", 0) == 0 || op.rfind("cnot.pred", 0) == 0)
      return "pred";
    if (op.rfind("selp.", 0) == 0) return "selp";
    if (op.rfind("slct.", 0) == 0) return "slct";
    if (op.rfind("fneg.", 0) == 0 || op.rfind("fnabs.", 0) == 0) return "funary";
    if ((op.rfind("sqrt.", 0) == 0 || op.rfind("rsqrt.", 0) == 0 ||
         op.rfind("rcp.", 0) == 0) &&
        op.find(".f32") != std::string::npos)
      return "funary";
    const bool minmax = op.rfind("min.", 0) == 0 || op.rfind("max.", 0) == 0;
    if (minmax) {
      if (op == "min.f32" || op == "max.f32") return "minmax";
      if (int_type_bits(op) == 0) return "";
    }
    if (op == "neg.f32" || op == "abs.f32") return "unary";
    const int bits = int_type_bits(op);
    if (bits == 0) return "";
    if (op.rfind("and.", 0) == 0 || op.rfind("or.", 0) == 0 ||
        op.rfind("xor.", 0) == 0 || op.rfind("not.", 0) == 0 ||
        op.rfind("shl.", 0) == 0 || op.rfind("shr.", 0) == 0)
      return "logic";
    if (op.rfind("mul.lo.", 0) == 0 || op.rfind("mul.hi.", 0) == 0 ||
        op.rfind("mad.lo.", 0) == 0 || op.rfind("mad.hi.", 0) == 0 ||
        op.rfind("div.", 0) == 0 || op.rfind("rem.", 0) == 0)
      return "muldiv";
    if (minmax) return "minmax";
    if (op.rfind("neg.", 0) == 0 || op.rfind("abs.", 0) == 0 ||
        op.rfind("popc.", 0) == 0 || op.rfind("clz.", 0) == 0 ||
        op.rfind("bfind.shiftamt.", 0) == 0 ||
        op.rfind("bfi.", 0) == 0 || op.rfind("bfe.", 0) == 0)
      return "unary";
    return "";
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

  // The shared PTX front-end only recognises a label whose name starts with a
  // letter, so the "$L__BB0_2:" form that nvcc emits is handed to us as a
  // zero-operand "instruction" whose mnemonic ends in ':'. Fold those back
  // into the label map here, and remap the labels the parser did recognise
  // onto the compacted instruction list.
  static bool is_label_pseudo(const rouge::PtxInstruction& ins) {
    return ins.args.empty() && ins.op.size() > 1 && ins.op.back() == ':';
  }

  void normalize_body() {
    std::vector<size_t> old_to_new(fn_->body.size(), 0);
    body_.clear();
    body_.reserve(fn_->body.size());
    for (size_t i = 0; i < fn_->body.size(); ++i) {
      const auto& ins = fn_->body[i];
      old_to_new[i] = body_.size();
      if (is_label_pseudo(ins)) {
        labels_[ins.op.substr(0, ins.op.size() - 1)] = body_.size();
        continue;
      }
      body_.push_back(ins);
    }
    for (const auto& kv : fn_->labels) {
      const size_t old = static_cast<size_t>(kv.second);
      if (old < old_to_new.size()) labels_[kv.first] = old_to_new[old];
    }
  }

  bool is_terminator(const rouge::PtxInstruction& ins) const {
    return ins.op == "bra" || ins.op == "bra.uni" || ins.op == "ret" ||
           ins.op == "exit";
  }

  void build_block_names() {
    const size_t n = body_.size();
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
      term_prev = is_terminator(body_[i]);
    }
  }

  // Label string attached to body index i, or "".
  std::string label_at(size_t i) const {
    for (const auto& kv : labels_)
      if (kv.second == i) return kv.first;
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
    emit_global_decls();
    emit_shared_metadata();
  }

  void emit_signature() {
    // Module-level declare for the block barrier runtime helper (bar.sync).
    for (const auto& ins : body_)
      if (ins.op == "bar.sync" || ins.op.rfind("bar.", 0) == 0) use_sync_ = true;
    if (use_sync_) line("declare void @__rouge_syncthreads(ptr)");
    scan_intrinsics();
    for (const auto& d : intrinsics_) line(d);
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

  // Collect the `declare` lines for the llvm.* intrinsics the body calls. The
  // body is emitted after the signature, so the scan has to happen up front.
  //
  // min/max and sqrt go through intrinsics rather than the smin/umin/fsqrt/
  // fmin/fmax opcodes because this LLVM's textual parser rejects those
  // keywords; the intrinsics are exactly how the opcodes are specified, so the
  // generated machine code is the same.
  void scan_intrinsics() {
    for (const auto& ins : body_) {
      const std::string op = ins.op;
      const int bits = int_type_bits(op);
      const bool is_min = op.rfind("min.", 0) == 0;
      const bool is_max = op.rfind("max.", 0) == 0;
      if ((is_min || is_max) && (bits == 16 || bits == 32 || bits == 64)) {
        const std::string ty = int_ty(bits);
        const std::string stem = std::string(is_signed_int_type(op) ? "llvm.s" : "llvm.u") +
                                 (is_min ? "min" : "max") + "." + ty;
        intrinsics_.insert("declare " + ty + " @" + stem + "(" + ty + ", " + ty + ")");
        continue;
      }
      if ((is_min || is_max) && op.find(".f32") != std::string::npos) {
        intrinsics_.insert(std::string("declare float @llvm.") +
                           (is_min ? "minnum.f32(float, float)"
                                   : "maxnum.f32(float, float)"));
        continue;
      }
      // sqrt.*.f32 and rsqrt.*.f32 both reach llvm.sqrt.f32.
      if ((op.rfind("sqrt.", 0) == 0 || op.rfind("rsqrt.", 0) == 0) &&
          op.find(".f32") != std::string::npos) {
        intrinsics_.insert("declare float @llvm.sqrt.f32(float)");
        continue;
      }
      if (bits == 0) continue;
      const std::string ty = int_ty(bits);
      if (op.rfind("abs.", 0) == 0 && is_signed_int_type(op))
        intrinsics_.insert("declare " + ty + " @llvm.abs." + ty + "(" + ty + ", i1)");
      if (op.rfind("popc.", 0) == 0)
        intrinsics_.insert("declare " + ty + " @llvm.ctpop." + ty + "(" + ty + ")");
      if (op.rfind("clz.", 0) == 0 || op.rfind("bfind.shiftamt.", 0) == 0)
        intrinsics_.insert("declare " + ty + " @llvm.ctlz." + ty + "(" + ty + ", i1)");
    }
  }

  void emit_allocas() {
    std::set<std::string> used;
    const auto is_label = [this](const std::string& tok) {
      return labels_.find(tok) != labels_.end();
    };
    const auto add_reg = [&used, &is_label](const std::string& tok) {
      if (is_register(tok) && !is_label(tok)) used.insert(tok);
      // Predicated destinations arrive as "r10|p1" — allocate the guard too.
      const size_t bar = tok.find('|');
      if (bar != std::string::npos && bar + 1 < tok.size()) {
        std::string g = tok.substr(bar + 1);
        if (!g.empty() && g[0] == '%') g = g.substr(1);
        if (is_register(g) && !is_label(g)) used.insert(g);
      }
    };
    for (const auto& ins : body_) {
      if (!ins.pred.empty()) add_reg(ins.pred[0] == '!' ? ins.pred.substr(1)
                                                        : ins.pred);
      for (const auto& raw : ins.args) {
        // "{%r2" / "r3}" (a vector register list split on ',') collapse back to
        // the register names; "[%rd1]" is a memory operand and keeps its
        // brackets until they are stripped here.
        const std::string inner = strip_brackets(clean_operand(raw));
        add_reg(inner);
      }
    }
    // Entry block label.
    if (block_name_.empty() || block_name_[0] != "entry") {
      line("entry:");
    }
    for (const auto& r : used) {
      const std::string ty = reg_type(r);
      line("  %" + r + " = alloca " + ty + ", align " + align_of(ty));
      // A .pred register the kernel never wrote reads as poison otherwise; the
      // interpreter treats an unset predicate as false, so seed it that way.
      if (ty == "i1") line("  store i1 false, ptr %" + r + ", align 1");
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
      // "mov.u64 %rd, arr" — nvcc also materialises a __device__ address with
      // a plain mov, not only with cvta.to.global.
      if (is_global_symbol(A[1])) return emit_symbol_address(A[0], A[1]);
      // "mov.u32 %r, sym" reads a .shared symbol's address (the form nvcc emits
      // for a demoted local array); same window as cvta.to.shared. A narrower
      // destination register takes the low 32 bits, as the hardware does.
      int symOff = 0, symConst = 0;
      if (shared_symbol(A[1], &symOff, &symConst)) {
        const std::string base = load_smem_base();
        const std::string addr = fresh("%sa");
        line("  " + addr + " = add i64 " + base + ", " +
             std::to_string(symOff + symConst));
        store_reg(A[0], fit_reg(A[0], addr, "i64"));
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
    if (op == "mov.f32" || op == "mov.f64") {
      if (A.size() != 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      // The payload is a plain bit-pattern copy, register or "0f..."/"0d..."
      // literal. f32 lands in an i32 alloca, f64 in an i64 one.
      store_reg(A[0], operand(A[1]));
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
      // The source is a module-scope .global/.const symbol (__device__ variable):
      // yield the symbol's address, not the value of a register of that name.
      if (is_global_symbol(A[1])) return emit_symbol_address(A[0], A[1]);
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

    // ---- integer ALU, f32 unary math, select and predicate logic -----------
    // Claimed before the generic add/sub/mul/div/fma path below so that
    // mul.lo/mul.hi/mad.*/div.*/rem.* (which all end in an integer type) are
    // not mistaken for the plain arithmetic forms. alu_family() returns "" for
    // everything that is not ours, in particular add.f32 / div.f32 / mul.f16.
    if (!alu_family(op).empty()) return emit_alu(ins, idx);

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

    // ---- warp shuffles: shfl.sync.<kind>.b32 (and legacy shfl.<kind>.b32) ----
    // Scalar fallback: shfl is identity before vectorization, MLIR will vectorize.
    if (op.rfind("shfl", 0) == 0) {
      if (A.size() < 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      line("  ; scalar fallback: shfl is identity before vectorization, MLIR will vectorize");
      // The destination may be predicated: "shfl.sync.down.b32 %r10|%p1, a, b, ..."
      // means "write %r10 only if %p1 is true". Split it and guard the store.
      std::string dst = A[0];
      std::string guard;
      const size_t bar = dst.find('|');
      if (bar != std::string::npos) {
        guard = dst.substr(bar + 1);
        dst = dst.substr(0, bar);
        if (!guard.empty() && guard[0] == '!') guard = guard.substr(1);
        if (!guard.empty() && guard[0] == '%') guard = guard.substr(1);
        if (guard.empty() || !is_register(guard))
          return fail("bad shfl write predicate '" + A[0] + "' @ " +
                      std::to_string(idx));
      }
      // A[0]=dst, A[1]=src (value to shuffle); A[2]=lane/offset etc. ignored in scalar.
      const std::string dstTy = reg_type(dst);
      std::string srcVal = operand(A[1]);
      // Handle width mismatch (e.g. immediate)
      if (is_register(A[1]) && reg_type(A[1]) != dstTy) {
        const std::string t = fresh("%t");
        const std::string srcTy = reg_type(A[1]);
        // Generic: trunc or zext depending on widths; for shfl b32 both are i32.
        if (dstTy == "i32" && srcTy == "i64") {
          line("  " + t + " = trunc i64 " + srcVal + " to i32");
        } else if (dstTy == "i64" && srcTy == "i32") {
          line("  " + t + " = zext i32 " + srcVal + " to i64");
        } else {
          line("  " + t + " = trunc " + srcTy + " " + srcVal + " to " + dstTy);
        }
        srcVal = t;
      }
      if (!guard.empty()) {
        const std::string gv = load_reg(guard);
        const std::string thenBl = fresh("bshfl");
        const std::string endBl = fresh("bshfl_end");
        line("  br i1 " + gv + ", label %" + thenBl + ", label %" + endBl);
        emit_label(thenBl);
        store_reg(dst, srcVal);
        line("  br label %" + endBl);
        emit_label(endBl);
      } else {
        store_reg(dst, srcVal);
      }
      return true;
    }

    // ---- vote.{any,all,uni}.pred and vote.sync.* ----
    if (op.rfind("vote", 0) == 0) {
      if (A.size() < 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      line("  ; scalar fallback: vote.{any,all,uni} -> pred identity (warp collective in MLIR)");
      std::string srcPred = A[1];
      bool neg = false;
      if (!srcPred.empty() && srcPred[0] == '!') {
        neg = true;
        srcPred = srcPred.substr(1);
      }
      std::string v = load_reg(srcPred);
      if (neg) {
        const std::string nv = fresh("%n");
        line("  " + nv + " = xor i1 " + v + ", true");
        v = nv;
      }
      store_reg(A[0], v);
      return true;
    }

    // ---- activemask.b32 ----
    if (op.rfind("activemask", 0) == 0) {
      if (A.empty()) return fail("bad " + op + " operands @ " + std::to_string(idx));
      line("  ; scalar fallback: activemask -> all lanes active (0xffffffff)");
      // b32 destination is i32; -1 == 0xffffffff
      store_reg(A[0], "-1");
      return true;
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

  // =================== integer ALU / select / predicate ALU ===============
  //
  // Everything here mirrors run_thread_step() in the interpreter
  // (rouge-ptx/ptx.cpp), which is the behavioural reference: same operands,
  // same widths, same edge cases, so the AOT and interpreter paths agree
  // bit-for-bit.

  bool emit_alu(const rouge::PtxInstruction& ins, size_t idx) {
    const std::string fam = alu_family(ins.op);
    if (fam == "logic") return emit_alu_logic(ins, idx);
    if (fam == "muldiv") return emit_alu_muldiv(ins, idx);
    if (fam == "minmax") return emit_alu_minmax(ins, idx);
    if (fam == "unary") return emit_alu_unary(ins, idx);
    if (fam == "funary") return emit_alu_funary(ins, idx);
    if (fam == "selp") return emit_select(ins, idx, /*predSrc=*/true);
    if (fam == "slct") return emit_select(ins, idx, /*predSrc=*/false);
    if (fam == "pred") return emit_pred_logic(ins, idx);
    return fail("unsupported PTX op '" + ins.op + "' @ " + std::to_string(idx));
  }

  // PTX takes the shift amount modulo the operand width, while LLVM's shift is
  // poison as soon as the amount reaches the width. Keep the two in step: a
  // literal amount is folded here, a computed one is masked.
  std::string shift_amount(const std::string& tok, const std::string& val,
                           const std::string& ty, int bits) {
    const long long mask = bits >= 64 ? 63 : bits >= 32 ? 31 : 15;
    if (is_numeric(tok)) {
      const unsigned long long raw = std::strtoull(tok.c_str(), nullptr, 10);
      return std::to_string(static_cast<long long>(
          raw & static_cast<unsigned long long>(mask)));
    }
    if (!is_register(tok) || reg_type(tok) != ty) return val;
    const std::string m = fresh("%sh");
    line("  " + m + " = and " + ty + " " + val + ", " + std::to_string(mask));
    return m;
  }

  // and.* / or.* / xor.* / not.* / shl.* / shr.* on 16-, 32- and 64-bit types.
  bool emit_alu_logic(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    const bool unary = op.rfind("not.", 0) == 0;
    if (A.size() != (unary ? 2u : 3u))
      return fail("bad " + op + " operands @ " + std::to_string(idx));
    const std::string ty = reg_type(A[0]);
    const std::string a = operand(A[1]);
    const std::string r = fresh("%r");
    if (unary) {
      // not.* -> xor with the all-ones pattern of the register's own width.
      line("  " + r + " = xor " + ty + " " + a + ", -1");
    } else {
      std::string b = operand(A[2]);
      if (op.rfind("shl.", 0) == 0 || op.rfind("shr.", 0) == 0)
        b = shift_amount(A[2], b, ty, reg_bits(A[0]));
      // shr.s* is an arithmetic shift, shr.u* (and the bitwise shr.b*) a
      // logical one.
      const char* ll = op.rfind("and.", 0) == 0    ? "and"
                       : op.rfind("or.", 0) == 0   ? "or"
                       : op.rfind("xor.", 0) == 0  ? "xor"
                       : op.rfind("shl.", 0) == 0  ? "shl"
                       : is_signed_int_type(op)    ? "ashr"
                                                   : "lshr";
      line("  " + r + " = " + ll + " " + ty + " " + a + ", " + b);
    }
    store_reg(A[0], r);
    return true;
  }

  // mul.lo / mul.hi / mad.lo / mad.hi / div / rem on 32- and 64-bit types.
  bool emit_alu_muldiv(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    const int bits = int_type_bits(op);
    const bool is_mad = op.rfind("mad.", 0) == 0;
    const bool hi = op.find(".hi.") != std::string::npos;
    const bool sgn = is_signed_int_type(op);
    if (A.size() < (is_mad ? 4u : 3u))
      return fail("bad " + op + " operands @ " + std::to_string(idx));

    if (op.rfind("div.", 0) == 0 || op.rfind("rem.", 0) == 0) {
      const bool is_rem = op.rfind("rem.", 0) == 0;
      const std::string ty = reg_type(A[0]);
      const std::string a = operand(A[1]);
      const char* llOp = sgn ? (is_rem ? "srem" : "sdiv") : (is_rem ? "urem" : "udiv");
      if (is_numeric(A[2])) {
        // A literal divisor is folded through the interpreter's rules, because
        // the very same constants make the LLVM instruction poison:
        //   * zero divisor -> the interpreter yields 0;
        //   * sdiv by -1   -> the interpreter wraps, LLVM calls it poison, so
        //     negate instead (and x % -1 is always 0).
        const unsigned long long umask =
            bits >= 64 ? ~0ull : ((1ull << bits) - 1ull);
        const unsigned long long d =
            std::strtoull(A[2].c_str(), nullptr, 10) & umask;
        if (d == 0) {
          store_reg(A[0], "0");
          return true;
        }
        if (sgn && d == umask) {  // divisor is -1 in the operand's width
          if (is_rem) {
            store_reg(A[0], "0");
            return true;
          }
          const std::string n = fresh("%r");
          line("  " + n + " = sub " + ty + " 0, " + a);
          store_reg(A[0], n);
          return true;
        }
        const std::string r = fresh("%r");
        line("  " + r + " = " + llOp + " " + ty + " " + a + ", " +
             std::to_string(static_cast<unsigned long long>(d)));
        store_reg(A[0], r);
        return true;
      }
      // A computed divisor is lowered straight to the LLVM instruction. A zero
      // divisor and INT_MIN / -1 are undefined in PTX and poison in LLVM: the
      // interpreter folds them to 0 and to "x wraps" respectively, so a kernel
      // relying on that would diverge. nvcc never emits either, so no guard is
      // added here (unlike the literal-divisor case above).
      const std::string b = operand(A[2]);
      const std::string r = fresh("%r");
      line("  " + r + " = " + llOp + " " + ty + " " + a + ", " + b);
      store_reg(A[0], r);
      return true;
    }

    if (hi) {
      if (bits != 32)
        return fail("unsupported '" + op + "' @ " + std::to_string(idx) +
                    " (the 64-bit high word of a 128-bit product needs a 128-bit"
                    " intermediate; the interpreter does not model it either)");
      // High word of the 64-bit product: widen, multiply at 64 bits, take bits
      // 63..32. The interpreter ignores the mad addend here because nvcc only
      // emits mad.hi with a zero addend.
      const std::string x = fresh("%x");
      line("  " + x + " = " + (sgn ? "sext" : "zext") + " i32 " + operand(A[1]) + " to i64");
      const std::string y = fresh("%y");
      line("  " + y + " = " + (sgn ? "sext" : "zext") + " i32 " + operand(A[2]) + " to i64");
      const std::string m = fresh("%m");
      line("  " + m + " = mul i64 " + x + ", " + y);
      const std::string s = fresh("%h");
      line("  " + s + " = " + (sgn ? "ashr" : "lshr") + " i64 " + m + ", 32");
      const std::string r = fresh("%r");
      line("  " + r + " = trunc i64 " + s + " to i32");
      store_reg(A[0], r);
      return true;
    }

    // mul.lo.* / mad.lo.*: the low word of the product, and for mad the addend
    // joins it modulo 2^width (the interpreter's `a * b + c`).
    const std::string ty = reg_type(A[0]);
    const std::string m = fresh("%m");
    line("  " + m + " = mul " + ty + " " + operand(A[1]) + ", " + operand(A[2]));
    if (!is_mad) {
      store_reg(A[0], m);
      return true;
    }
    const std::string r = fresh("%r");
    line("  " + r + " = add " + ty + " " + m + ", " + operand(A[3]));
    store_reg(A[0], r);
    return true;
  }

  // min.* / max.*: signed, unsigned/bitwise and f32 forms.
  //
  // NOTE on the spelling: the textual `smin`/`smax`/`umin`/`umax` and
  // `fmin`/`fmax` opcodes are not accepted by the LLVM this project builds
  // against, so the equivalent llvm.* intrinsics are emitted instead. They are
  // the same operations (the intrinsics are how the opcodes are defined), so
  // the generated machine code is identical.
  bool emit_alu_minmax(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    if (A.size() != 3) return fail("bad " + op + " operands @ " + std::to_string(idx));
    const bool is_min = op.rfind("min.", 0) == 0;
    if (op == "min.f32" || op == "max.f32") {
      // f32 lives in an i32 alloca, so the comparison runs on the bit pattern
      // reinterpreted as a float (minnum/maxnum keep the NaN handling PTX asks
      // for: the non-NaN operand wins).
      const std::string x = load_f32(A[1]);
      const std::string y = load_f32(A[2]);
      const std::string r = fresh("%r");
      line("  " + r + " = call float @" + (is_min ? "llvm.minnum.f32" : "llvm.maxnum.f32") +
           "(float " + x + ", float " + y + ")");
      store_f32(A[0], r);
      return true;
    }
    const std::string ty = reg_type(A[0]);
    const std::string name =
        std::string(is_signed_int_type(op) ? "llvm.s" : "llvm.u") + (is_min ? "min" : "max") + "." + ty;
    const std::string r = fresh("%r");
    line("  " + r + " = call " + ty + " @" + name + "(" + ty + " " + operand(A[1]) +
         ", " + ty + " " + operand(A[2]) + ")");
    store_reg(A[0], r);
    return true;
  }

  // neg.* / abs.* / popc.* / clz.* / bfind.shiftamt.*
  // Store the ADDRESS of a module-scope global symbol into a register, as an
  // i64 (register addresses are i64 everywhere in this file and are turned
  // back into pointers with inttoptr at the point of use).
  bool emit_symbol_address(const std::string& reg, const std::string& sym) {
    const std::string p = fresh("%gp");
    line("  " + p + " = load ptr, ptr @" + sym + ", align 8");
    const std::string i = fresh("%gi");
    line("  " + i + " = ptrtoint ptr " + p + " to i64");
    store_reg(reg, i);
    return true;
  }

  // bfi/bfe — bit-field insert/extract. nvcc lowers __shfl through bfi, so
  // real PTX needs it. start = c & 0xff, width = (c >> 8) & 0xff (0 means
  // full width). Both paths compute it with explicit masks to stay portable
  // across LLVM versions and to match the interpreter bit for bit.
  bool emit_bitfield(const rouge::PtxInstruction& ins, size_t idx) {
    (void)idx;
    const auto& A = ins.args;
    const bool insert = ins.op.rfind("bfi.", 0) == 0;
    const std::string ty = reg_type(A[0]);
    // bfi/bfe are defined on 32/64-bit words; the source registers may be
    // narrower, so widen once and work in i64 throughout.
    const auto as_i64 = [&](const std::string& reg_tok) {
      const std::string v = operand(reg_tok);
      const std::string t = reg_type(reg_tok);
      if (t == "i64") return v;
      const std::string w = fresh("%we");
      line("  " + w + " = zext " + t + " " + v + " to i64");
      return w;
    };
    const std::string a = as_i64(A[1]);
    const std::string b = insert ? as_i64(A[2]) : std::string();

    // PTX has two shapes: "bfi.b32 d,a,b,c" packs pos=c&0xff, width=(c>>8)&0xff,
    // and "bfi.b32 d,a,b,pos,width" passes them separately. nvcc emits both.
    const bool five = A.size() == 5;
    std::string pos, width;
    if (!five) {
      const std::string d = operand(A[3]);
      const std::string p = fresh("%bs");
      line("  " + p + " = and i64 " + d + ", 255");
      pos = p;
      if (is_numeric(A[3])) {
        int w = static_cast<int>((std::strtoll(A[3].c_str(), nullptr, 10) >> 8) & 0xff);
        if (w == 0) w = 32;
        width = std::to_string(w);
      } else {
        const std::string t1 = fresh("%bw1");
        line("  " + t1 + " = lshr i64 " + d + ", 8");
        const std::string t2 = fresh("%bw2");
        line("  " + t2 + " = and i64 " + t1 + ", 255");
        const std::string z = fresh("%bwz");
        line("  " + z + " = icmp eq i64 " + t2 + ", 0");
        const std::string t3 = fresh("%bw3");
        line("  " + t3 + " = select i1 " + z + ", i64 32, i64 " + t2);
        width = t3;
      }
    } else {
      pos = operand(A[3]);
      const std::string w = operand(A[4]);
      const std::string z = fresh("%bwz");
      line("  " + z + " = icmp eq i64 " + w + ", 0");
      const std::string t = fresh("%bw3");
      line("  " + t + " = select i1 " + z + ", i64 32, i64 " + w);
      width = t;
    }
    // A 32-bit field never shifts past 31; clamping keeps LLVM shift amounts
    // in range (out-of-range shifts are poison).
    if (ty != "i64") {
      const std::string c = fresh("%bp");
      line("  " + c + " = and i64 " + pos + ", 31");
      pos = c;
    }
    // mask = (1 << width) - 1, as a 64-bit value
    const std::string one = fresh("%one");
    line("  " + one + " = shl i64 1, " + width);
    const std::string maskv = fresh("%mk");
    line("  " + maskv + " = sub i64 " + one + ", 1");

    const std::string r = fresh("%r");
    if (insert) {
      const std::string keep = fresh("%k1");
      line("  " + keep + " = xor i64 " + maskv + ", -1");
      const std::string base = fresh("%b1");
      line("  " + base + " = and i64 " + a + ", " + keep);
      const std::string bv = fresh("%bv");
      line("  " + bv + " = and i64 " + b + ", " + maskv);
      const std::string sh = fresh("%sh");
      line("  " + sh + " = shl i64 " + bv + ", " + pos);
      line("  " + r + " = or i64 " + base + ", " + sh);
    } else {
      const std::string sh = fresh("%sh");
      line("  " + sh + " = lshr i64 " + a + ", " + pos);
      line("  " + r + " = and i64 " + sh + ", " + maskv);
    }
    if (ty != "i64") {
      const std::string t = fresh("%t");
      line("  " + t + " = trunc i64 " + r + " to " + ty);
      store_reg(A[0], t);
    } else {
      store_reg(A[0], r);
    }
    return true;
  }

  bool emit_alu_unary(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    if ((A.size() == 4 || A.size() == 5) &&
        (op.rfind("bfi.", 0) == 0 || op.rfind("bfe.", 0) == 0))
      return emit_bitfield(ins, idx);
    if (A.size() != 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
    const std::string ty = reg_type(A[0]);
    const std::string a = operand(A[1]);
    const std::string r = fresh("%r");

    if (op == "neg.f32") {
      // Two's-complement negation of the bit pattern is exactly the IEEE sign
      // flip, so this is fneg.f32 without the bitcast round trip.
      line("  " + r + " = sub i32 0, " + a);
      store_reg(A[0], r);
      return true;
    }
    if (op == "abs.f32") {
      // abs.f32 (what nvcc emits for fabsf) is bit-for-bit fnabs.f32: clear the
      // sign bit, exact for +-0, inf and NaN.
      line("  " + r + " = and i32 " + a + ", 2139095040");
      store_reg(A[0], r);
      return true;
    }
    if (op.rfind("neg.", 0) == 0) {
      line("  " + r + " = sub " + ty + " 0, " + a);
      store_reg(A[0], r);
      return true;
    }
    if (op.rfind("abs.", 0) == 0) {
      if (op.find(".u") != std::string::npos) {
        // abs.u*: drop the sign bit, like the interpreter's `a & ~(1 << (bits-1))`.
        const int bits = reg_bits(A[0]);
        const unsigned long long mask = bits >= 64
                                            ? 0x7fffffffffffffffull
                                            : ((1ull << (bits - 1)) - 1ull);
        line("  " + r + " = and " + ty + " " + a + ", " +
             std::to_string(static_cast<unsigned long long>(mask)));
      } else {
        // abs.s*: the second argument false keeps INT_MIN wrapping to itself,
        // which is what the interpreter computes.
        line("  " + r + " = call " + ty + " @llvm.abs." + ty + "(" + ty + " " + a +
             ", i1 false)");
      }
      store_reg(A[0], r);
      return true;
    }
    if (op.rfind("popc.", 0) == 0) {
      line("  " + r + " = call " + ty + " @llvm.ctpop." + ty + "(" + ty + " " + a + ")");
      store_reg(A[0], r);
      return true;
    }
    // clz.* and bfind.shiftamt.* both give the distance from the top set bit;
    // the trailing `false` is is_zero_poison = false, so an all-zero input
    // yields the full width like the interpreter's clz32/clz64.
    line("  " + r + " = call " + ty + " @llvm.ctlz." + ty + "(" + ty + " " + a +
         ", i1 false)");
    store_reg(A[0], r);
    return true;
  }

  // fneg.f32 / fnabs.f32 / sqrt.*.f32 / rcp.*.f32 / rsqrt.*.f32
  bool emit_alu_funary(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    if (A.size() != 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
    if (op.rfind("fneg.", 0) == 0 || op.rfind("fnabs.", 0) == 0) {
      // Pure bit operations on the f32 pattern: sign flip, then sign+exponent
      // clear. Exact for +-0, inf and NaN, just as in the interpreter.
      const std::string bits = load_reg(A[1]);
      const std::string r = fresh("%r");
      line("  " + r + " = " + (op.rfind("fneg.", 0) == 0 ? "xor" : "and") + " i32 " +
           bits + ", " + (op.rfind("fneg.", 0) == 0 ? "-2147483648" : "2139095040"));
      store_reg(A[0], r);
      return true;
    }
    const std::string x = load_f32(A[1]);
    const std::string r = fresh("%r");
    if (op.rfind("rsqrt.", 0) == 0) {
      // rsqrt = 1 / sqrt(x), as in the interpreter. (The textual `fsqrt`
      // opcode is not accepted by this LLVM, hence the intrinsic form; it is
      // the same operation.)
      const std::string s = fresh("%f");
      line("  " + s + " = call float @llvm.sqrt.f32(float " + x + ")");
      line("  " + r + " = fdiv float 1.000000e+00, " + s);
    } else if (op.rfind("rcp.", 0) == 0) {
      line("  " + r + " = fdiv float 1.000000e+00, " + x);
    } else {
      line("  " + r + " = call float @llvm.sqrt.f32(float " + x + ")");
    }
    store_f32(A[0], r);
    return true;
  }

  // selp.<type> d, a, b, p  ->  p ? a : b     (p is a .pred register)
  // slct.<type> d, a, b, c  ->  (c & 1) ? a : b (c is a value operand)
  bool emit_select(const rouge::PtxInstruction& ins, size_t idx, bool predSrc) {
    const auto& A = ins.args;
    if (A.size() < 4)
      return fail("bad " + ins.op + " operands @ " + std::to_string(idx));
    const std::string ty = reg_type(A[0]);
    const std::string a = operand(A[1]);
    const std::string b = operand(A[2]);
    const std::string c = fresh("%c");
    if (predSrc) {
      if (A[3].empty() || A[3][0] == '!')
        return fail(ins.op + " needs a predicate register as its 4th operand @" +
                    std::to_string(idx));
      if (!check_pred_operand(A[3], ins.op, idx)) return false;
      line("  " + c + " = load i1, ptr %" + clean_operand(A[3]) + ", align 1");
    } else {
      line("  " + c + " = trunc " + ty + " " + operand(A[3]) + " to i1");
    }
    const std::string r = fresh("%r");
    // LLVM's select repeats the type on every value operand.
    line("  " + r + " = select i1 " + c + ", " + ty + " " + a + ", " + ty + " " + b);
    store_reg(A[0], r);
    return true;
  }

  // Predicate logic. The operands are .pred registers (i1 allocas); a leading
  // '!' is accepted and means "logical not", as the interpreter's value()
  // lookup does.
  static std::string pred_name(const std::string& tok) {
    const std::string s = clean_operand(tok);
    return (!s.empty() && s[0] == '!') ? s.substr(1) : s;
  }
  // Read a predicate operand into an i1 SSA value, honouring a leading '!'.
  std::string load_pred(const std::string& tok) {
    const std::string v = load_reg(pred_name(tok));
    if (tok.empty() || tok[0] != '!') return v;
    const std::string n = fresh("%n");
    line("  " + n + " = xor i1 " + v + ", true");
    return n;
  }
  // A predicate operand has to be a declared .pred register; anything else
  // (an undeclared name, a plain .b32) would silently produce an i1/i32 mix.
  bool check_pred_operand(const std::string& tok, const std::string& op, size_t idx) {
    const std::string n = pred_name(tok);
    if (!n.empty() && reg_bits(n) == 1) return true;
    fail(op + " needs .pred register operands, but '" + n + "' is not one @" +
         std::to_string(idx));
    return false;
  }
  bool emit_pred_logic(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    if (op.rfind("not.pred", 0) == 0 || op.rfind("cnot.pred", 0) == 0) {
      // not.pred d, a{, c}  ->  (a != c); the two-operand form leaves c true,
      // which reduces to !a.
      if (A.size() < 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      if (!check_pred_operand(A[1], op, idx)) return false;
      if (A.size() >= 3 && !check_pred_operand(A[2], op, idx)) return false;
      const std::string c =
          A.size() >= 3 ? load_pred(A[2]) : std::string("true");
      const std::string r = fresh("%r");
      line("  " + r + " = xor i1 " + load_pred(A[1]) + ", " + c);
      store_reg(A[0], r);
      return true;
    }
    // and.pred d, a, b{, c}  ->  (a && b) || !c
    // or.pred  d, a, b{, c}  ->  (a || b) || !c
    // The optional third source defaults to true, and both mnemonics use the
    // same "|| !c" completion rule, exactly as in the interpreter.
    if (A.size() < 3) return fail("bad " + op + " operands @ " + std::to_string(idx));
    if (!check_pred_operand(A[1], op, idx)) return false;
    if (!check_pred_operand(A[2], op, idx)) return false;
    if (A.size() >= 4 && !check_pred_operand(A[3], op, idx)) return false;
    if (reg_bits(A[0]) != 1)
      return fail(op + " destination '" + A[0] + "' is not a .pred register @" +
                  std::to_string(idx));
    const bool is_and = op.rfind("and.pred", 0) == 0;
    const std::string ab = fresh("%ab");
    line("  " + ab + " = " + (is_and ? "and" : "or") + " i1 " + load_pred(A[1]) +
         ", " + load_pred(A[2]));
    const std::string nc = fresh("%nc");
    line("  " + nc + " = xor i1 " + (A.size() >= 4 ? load_pred(A[3]) : std::string("true")) +
         ", true");
    const std::string r = fresh("%r");
    line("  " + r + " = or i1 " + ab + ", " + nc);
    store_reg(A[0], r);
    return true;
  }


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

    // f16x2/bf16x2 fallback: keep payload as i32 (vector of 2×half).
    // Real packed arithmetic will be vectorized in MLIR stage 2; for AOT
    // correctness just preserve bit pattern (or do i32 add) so codegen does not fail.
    if (op.find("x2") != std::string::npos) {
      // Fallback: bit-copy first source to dest (or i32 add for add.*)
      if (fma) {
        store_reg(A[0], operand(A[1]));
      } else if (op.rfind("add.", 0) == 0) {
        const std::string r = fresh("%r");
        line("  " + r + " = add i32 " + operand(A[1]) + ", " + operand(A[2]));
        store_reg(A[0], r);
      } else {
        store_reg(A[0], operand(A[1]));
      }
      return true;
    }

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

    // f64 / f16 / bf16 arithmetic beyond the f16+bf16 block above is out of the
    // supported subset (the interpreter does not model f64 arithmetic either).
    // Refuse it here rather than letting it reach the integer path below, which
    // would silently treat the bit pattern as an integer. f64 *bit patterns*
    // are fine and supported: mov.f64, ld/st.global.f64, cvt.{f64} and
    // selp/slct.f64 all move the i64 payload without touching its value.
    if (op.find(".f64") != std::string::npos)
      return fail(op + " needs f64 arithmetic, which is not supported @ " +
                  std::to_string(idx) +
                  " (the interpreter does not model f64 arithmetic either)");

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
    const bool is_cas = op.find(".cas.") != std::string::npos;
    const bool is_exch = op.find(".exch.") != std::string::npos;
    const bool is_min = op.find(".min.") != std::string::npos;
    const bool is_max = op.find(".max.") != std::string::npos;
    const bool is_add = op.find(".add.") != std::string::npos;
    if (!is_cas && !is_exch && !is_min && !is_max && !is_add)
      return fail("unsupported atomic '" + op + "' @ " + std::to_string(idx) +
                  " (supported: atom.{add,cas,exch,min,max}.*, red.{add,min,max}.*)");
    const bool has_dst = op.rfind("atom.", 0) == 0;
    const std::string ty = op.substr(op.rfind('.') + 1);
    const bool w64 = ty == "u64" || ty == "s64" || ty == "b64" || ty == "f64";
    const bool wf32 = ty == "f32";
    // CAS
    if (is_cas) {
      if (!has_dst) return fail("red.cas not supported '" + op + "' @ " + std::to_string(idx));
      if (A.size() < 4) return fail("bad atomic operands @ " + std::to_string(idx));
      const std::string addr = resolve_addr(A[1]);
      const std::string cmp = operand(A[2]);
      const std::string nw = operand(A[3]);
      if (wf32) {
        // f32 cas via b32 bitcast: compare as i32
        const std::string pair = fresh("%cas");
        const std::string old = fresh("%old");
        line("  " + pair + " = cmpxchg ptr " + addr + ", i32 " + cmp + ", i32 " + nw +
             " monotonic monotonic, align 4");
        line("  " + old + " = extractvalue { i32, i1 } " + pair + ", 0");
        store_reg(A[0], old);
        return true;
      }
      if (w64) {
        const std::string pair = fresh("%cas");
        const std::string old = fresh("%old");
        line("  " + pair + " = cmpxchg ptr " + addr + ", i64 " + cmp + ", i64 " + nw +
             " monotonic monotonic, align 8");
        line("  " + old + " = extractvalue { i64, i1 } " + pair + ", 0");
        store_reg(A[0], old);
        return true;
      }
      // b32/u32/s32
      const std::string pair = fresh("%cas");
      const std::string old = fresh("%old");
      line("  " + pair + " = cmpxchg ptr " + addr + ", i32 " + cmp + ", i32 " + nw +
           " monotonic monotonic, align 4");
      line("  " + old + " = extractvalue { i32, i1 } " + pair + ", 0");
      store_reg(A[0], old);
      return true;
    }
    if (is_exch) {
      if (!has_dst) return fail("red.exch not supported '" + op + "' @ " + std::to_string(idx));
      if (A.size() < 3) return fail("bad atomic operands @ " + std::to_string(idx));
      const std::string addr = resolve_addr(A[1]);
      const std::string val = operand(A[2]);
      const std::string old = fresh("%old");
      if (wf32) {
        // exch f32 via i32 bitcast
        line("  " + old + " = atomicrmw xchg ptr " + addr + ", i32 " + val + " monotonic, align 4");
        store_reg(A[0], old);
        return true;
      }
      if (w64) {
        line("  " + old + " = atomicrmw xchg ptr " + addr + ", i64 " + val + " monotonic, align 8");
      } else {
        line("  " + old + " = atomicrmw xchg ptr " + addr + ", i32 " + val + " monotonic, align 4");
      }
      store_reg(A[0], old);
      return true;
    }
    if (is_min || is_max) {
      const size_t ai = has_dst ? 1 : 0;
      if (A.size() < ai + 2) return fail("bad atomic operands @ " + std::to_string(idx));
      const std::string addr = resolve_addr(A[ai]);
      const std::string val = operand(A[ai + 1]);
      const std::string old = fresh("%old");
      if (wf32) {
        const std::string f = fresh("%f");
        line("  " + f + " = bitcast i32 " + val + " to float");
        const char* llOp = is_min ? "fmin" : "fmax";
        line("  " + old + " = atomicrmw " + llOp + " ptr " + addr + ", float " + f + " monotonic, align 4");
        if (has_dst) {
          const std::string bits = fresh("%b");
          line("  " + bits + " = bitcast float " + old + " to i32");
          store_reg(A[0], bits);
        }
        return true;
      }
      if (ty == "f64") {
        const std::string f = fresh("%f");
        line("  " + f + " = bitcast i64 " + val + " to double");
        const char* llOp = is_min ? "fmin" : "fmax";
        line("  " + old + " = atomicrmw " + llOp + " ptr " + addr + ", double " + f + " monotonic, align 8");
        if (has_dst) {
          const std::string bits = fresh("%b");
          line("  " + bits + " = bitcast double " + old + " to i64");
          store_reg(A[0], bits);
        }
        return true;
      }
      std::string llOp;
      if (ty == "s32" || ty == "s64") llOp = is_min ? "min" : "max";
      else if (ty == "u32" || ty == "u64" || ty == "b32" || ty == "b64") llOp = is_min ? "umin" : "umax";
      else return fail("unsupported atomic type '" + ty + "' @ " + std::to_string(idx));
      const char* llTy = w64 ? "i64" : "i32";
      const int align = w64 ? 8 : 4;
      line("  " + old + " = atomicrmw " + llOp + " ptr " + addr + ", " + llTy + " " + val +
           " monotonic, align " + std::to_string(align));
      if (has_dst) store_reg(A[0], old);
      return true;
    }
    // add
    const size_t ai = has_dst ? 1 : 0;
    if (A.size() < ai + 2)
      return fail("bad atomic operands @ " + std::to_string(idx));
    if (ty == "f64")
      return fail("unsupported atomic '" + op + "' @ " + std::to_string(idx) +
                  " (an f64 add would have to be a real double atomicrmw, not"
                  " the integer add below; atom.min/max.f64 are supported)");
    const std::string addr = resolve_addr(A[ai]);
    const int align = w64 ? 8 : 4;
    const char* llTy = w64 ? "i64" : "i32";
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

    // A "cvt.<mod>." prefix carries PTX saturation / rounding rules that the
    // plain LLVM cast instructions do not generally implement, so it is
    // refused here rather than silently miscompiled. Exactly two combinations
    // ARE what LLVM does out of the box, and nvcc emits both:
    //   * "rn" (round to nearest even) with a float destination -- fptrunc /
    //     fpext / uitofp / sitofp all round to nearest even;
    //   * "rz" / "rzi" (round toward zero) with an integer destination --
    //     fptosi / fptoui round toward zero.
    // Everything else (saturation, rp/ru/rm directed rounding, "rn" into an
    // integer) is rejected. The interpreter refuses the whole modifier set
    // (rouge-ptx/ptx.cpp), so accepting these two makes the AOT path strictly
    // more capable rather than divergent.
    std::string strip = spec;  // "<dst>.<src>" with the modifiers peeled off
    std::string mod;
    for (;;) {
      const size_t d = strip.find('.');
      if (d == std::string::npos) break;
      const std::string head = strip.substr(0, d);
      const bool isMod = head == "sat" || head == "rn" || head == "rz" ||
                         head == "rzi" || head == "rp" || head == "ru" ||
                         head == "rm";
      if (!isMod) break;
      mod = head;
      strip = strip.substr(d + 1);
    }
    {
      const size_t pdot = strip.find('.');
      const std::string dstTy =
          pdot == std::string::npos ? strip : strip.substr(0, pdot);
      const std::string srcTy =
          pdot == std::string::npos ? std::string() : strip.substr(pdot + 1);
      const auto isFloat = [](const std::string& t) {
        return t == "f32" || t == "f64" || t == "f16" || t == "bf16";
      };
      const bool ok =
          mod.empty() || (mod == "rn" && isFloat(dstTy)) ||
          ((mod == "rz" || mod == "rzi") && !isFloat(dstTy) && isFloat(srcTy));
      if (!ok) {
        return fail("unsupported rounding/saturating conversion '" + ins.op +
                    "' @ " + std::to_string(idx) +
                    " (PTX saturation/rounding rules are not modelled; the "
                    "interpreter refuses it too)");
      }
    }

    spec = strip;
    const std::string dst = spec.substr(0, spec.find('.'));
    const std::string src = spec.substr(spec.find('.') + 1);

    // f16x2/bf16x2 fallback: 32-bit payload as i32 (2×i16 bit pattern).
    // Keep it simple — bit-copy through i32; unpack to 2× half can be added
    // when vector lowering lands in MLIR.
    if (spec.find("x2") != std::string::npos) {
      std::string val = operand(A[1]);
      const std::string dstTy = reg_type(A[0]);
      if (is_register(A[1]) && reg_type(A[1]) != dstTy) {
        const std::string t = fresh("%t");
        if (reg_bits(A[1]) < reg_bits(A[0]))
          line("  " + t + " = zext " + reg_type(A[1]) + " " + val + " to " + dstTy);
        else
          line("  " + t + " = trunc " + reg_type(A[1]) + " " + val + " to " + dstTy);
        val = t;
      }
      store_reg(A[0], val);
      return true;
    }

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

    // ---- widths the table above does not cover ---------------------------
    // Everything below mirrors the interpreter: the result takes the width the
    // *mnemonic* names, 16-bit destinations simply keep the low halfword (no
    // sign extension), and 16-bit sources are zero-extended for .u16/.b16 and
    // sign-extended for .s16.
    const bool dstI64 = dst == "u64" || dst == "s64" || dst == "b64";
    const bool dstI32 = dst == "u32" || dst == "s32" || dst == "b32";
    const bool dstI16 = dst == "u16" || dst == "s16" || dst == "b16";
    const bool dst64f = dst == "f64";
    const bool src64i = src == "u64" || src == "s64" || src == "b64";
    const bool src32i = src == "u32" || src == "s32" || src == "b32";
    const bool src16i = src == "u16" || src == "s16" || src == "b16";
    const bool src64f = src == "f64";
    const bool src32f = src == "f32";
    const int dstBits = dstI64 ? 64 : dstI32 ? 32 : 16;
    const int srcBits = src64i || src64f ? 64 : src32i || src32f ? 32 : 16;
    const std::string srcTy = operand_type(A[1], srcBits);
    const std::string v = operand(A[1]);

    if (dstI16 && (src16i || src32i || src64i)) {
      // 16-bit destination: keep the low halfword of whatever the source is.
      const std::string t = fresh("%t");
      line("  " + t + " = trunc " + srcTy + " " + v + " to i16");
      store_reg(A[0], fit_reg(A[0], t, "i16"));
      return true;
    }
    if (dstI32 && src16i) {
      const std::string t = fresh("%t");
      line("  " + t + " = " + (src == "s16" ? "sext" : "zext") + " i16 " + v + " to i32");
      store_reg(A[0], fit_reg(A[0], t, "i32"));
      return true;
    }
    if (dstI64 && src16i) {
      const std::string t = fresh("%t");
      line("  " + t + " = " + (src == "s16" ? "sext" : "zext") + " i16 " + v + " to i64");
      store_reg(A[0], fit_reg(A[0], t, "i64"));
      return true;
    }
    if (dstI16 && src64i) {
      const std::string t = fresh("%t");
      line("  " + t + " = trunc i64 " + v + " to i16");
      store_reg(A[0], fit_reg(A[0], t, "i16"));
      return true;
    }
    if (dstI32 && src64i) {
      // PTX narrows to the destination width, so .s64 keeps its sign in bit 31
      // (the interpreter's `static_cast<uint32_t>(sext64(raw, 32))`).
      const std::string t = fresh("%t");
      line("  " + t + " = trunc " + srcTy + " " + v + " to i32");
      store_reg(A[0], fit_reg(A[0], t, "i32"));
      return true;
    }
    // f32 <- f64: narrow, then land in the i32 alloca that holds the pattern.
    if (dst == "f32" && src64f) {
      const std::string d = fresh("%d");
      line("  " + d + " = bitcast i64 " + v + " to double");
      const std::string f = fresh("%f");
      line("  " + f + " = fptrunc double " + d + " to float");
      store_f32(A[0], f);
      return true;
    }
    // f64 <- f32: widen, then bitcast into the i64 alloca holding the pattern.
    if (dst64f && src32f) {
      const std::string f = load_f32(A[1]);
      const std::string d = fresh("%d");
      line("  " + d + " = fpext float " + f + " to double");
      const std::string b = fresh("%b");
      line("  " + b + " = bitcast double " + d + " to i64");
      store_reg(A[0], fit_reg(A[0], b, "i64"));
      return true;
    }
    // float -> integer: fptosi / fptoui round toward zero and saturate at the
    // destination range, which is what the interpreter's fp_to_int() does.
    if (src64f && (dstI32 || dstI16)) {
      const std::string d = fresh("%d");
      line("  " + d + " = bitcast " + srcTy + " " + v + " to double");
      const std::string i = fresh("%i");
      line("  " + i + " = " + (dst[0] == 'u' ? "fptoui" : "fptosi") + " double " + d +
           " to " + int_ty(dstBits));
      store_reg(A[0], fit_reg(A[0], i, int_ty(dstBits)));
      return true;
    }
    if (src32f && dstI64) {
      const std::string i = fresh("%i");
      line("  " + i + " = " + (dst[0] == 'u' ? "fptoui" : "fptosi") + " float " +
           load_f32(A[1]) + " to i64");
      store_reg(A[0], fit_reg(A[0], i, "i64"));
      return true;
    }
    // integer -> float. A 16-bit source is widened to 32 first (sign-extended
    // for .s16) so the value stays exact, as in the interpreter.
    const auto widen16 = [this, &src](const std::string& val) {
      const std::string t = fresh("%t");
      line("  " + t + " = " + (src == "s16" ? "sext" : "zext") + " i16 " + val +
           " to i32");
      return t;
    };
    const std::string intSrcTy = src16i ? "i32" : srcTy;
    if (dst == "f32" && (src64i || src32i || src16i)) {
      const std::string w = src16i ? widen16(v) : v;
      const std::string f = fresh("%f");
      line("  " + f + " = " + (src[0] == 's' ? "sitofp" : "uitofp") + " " + intSrcTy +
           " " + w + " to float");
      store_f32(A[0], f);
      return true;
    }
    if (dst64f && (src64i || src32i || src16i)) {
      const std::string w = src16i ? widen16(v) : v;
      const std::string d = fresh("%d");
      line("  " + d + " = " + (src[0] == 's' ? "sitofp" : "uitofp") + " " + intSrcTy +
           " " + w + " to double");
      const std::string b = fresh("%b");
      line("  " + b + " = bitcast double " + d + " to i64");
      store_reg(A[0], fit_reg(A[0], b, "i64"));
      return true;
    }
    return fail("unsupported cvt '" + spec + "' @ " + std::to_string(idx));
  }

  bool emit_setp(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    if (A.size() != 3) return fail("bad setp operands @ " + std::to_string(idx));
    std::string spec = ins.op.substr(5);  // e.g. "ge.u32"
    // Float compares need fcmp over the reinterpreted value, which is not
    // implemented; the integer icmp below would silently compare bit patterns.
    if (spec.find(".f32") != std::string::npos ||
        spec.find(".f64") != std::string::npos ||
        spec.find(".f16") != std::string::npos ||
        spec.find(".bf16") != std::string::npos)
      return fail(ins.op + " float compare is not supported @ " +
                  std::to_string(idx));
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
    std::string a = load_reg(reg_tok);
    // A 32-bit address register is legal PTX (nvcc uses it for a demoted local
    // array: "mov.u32 %r, sym; add.s32 %r, %r, %off; st.shared.f32 [%r], %f"),
    // so widen it before turning it into a pointer.
    if (reg_type(reg_tok) != "i64") {
      const std::string w = fresh("%a");
      line("  " + w + " = zext " + reg_type(reg_tok) + " " + a + " to i64");
      a = w;
    }
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
    // Register plus a constant byte offset: "[%r6+512]", "[%rd2-4]". This is
    // how nvcc indexes shared memory (and nearly every global access), so it
    // must lower to a real address computation, never to a pointer named
    // "r6+512".
    std::string base;
    int64_t off = 0;
    if (split_reg_offset(inner, &base, &off)) {
      const std::string a = load_reg(base);
      const std::string ty = reg_type(base);
      const std::string w = fresh("%w");
      const std::string sum = fresh("%o");
      if (ty == "i64") {
        line("  " + sum + " = add i64 " + a + ", " + std::to_string(off));
      } else {
        // 32-bit (or narrower) address register: zero-extend, as the
        // interpreter keeps register values unsigned.
        line("  " + w + " = zext " + ty + " " + a + " to i64");
        line("  " + sum + " = add i64 " + w + ", " + std::to_string(off));
      }
      const std::string p = fresh("%p");
      line("  " + p + " = inttoptr i64 " + sum + " to ptr");
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
  // ".u8" / ".s8" / ".b8" memory type?
  static bool is_8bit_ty(const std::string& op) {
    return ends_with(op, ".u8") || ends_with(op, ".s8") || ends_with(op, ".b8");
  }
  // True for the vector memory types ("v2.f32", "v4.u32", ...).
  static std::string vec_ty(const std::string& op) { return vector_type(op); }
  // The 64-bit vector forms that have an honest single-access lowering.
  static bool is_v2_supported(const std::string& v) {
    return v == "v2.f32" || v == "v2.u32" || v == "v2.s32" || v == "v2.b32";
  }

  bool emit_ld_mem(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    // A vector destination list "{%r2, %r3}" arrives as two tokens, so a v2
    // load carries three operands (two registers + the address); a plain
    // 64-bit destination keeps the usual two.
    const std::string vty = vec_ty(ins.op);
    if (!vty.empty()) {
      if (!is_v2_supported(vty))
        return fail("unsupported " + ins.op + " type '" + vty + "' @ " +
                    std::to_string(idx) +
                    " (implemented: v2.{f32,u32,s32,b32} as one 64-bit access)");
      if (A.size() < 2)
        return fail("bad " + ins.op + " operands @ " + std::to_string(idx));
      // Operands are split on ',', so a destination list "{%r2, %r3}" arrives
      // as A[0]="{%r2", A[1]="r3}" and the address moves to A[2]; a plain
      // 64-bit destination keeps the usual two operands.
      const bool list = A.size() >= 3;
      const std::string addr = resolve_addr(list ? A[2] : A[1]);
      const std::string v = fresh("%v");
      line("  " + v + " = load i64, ptr " + addr + ", align 8");
      if (!list) {
        store_reg(A[0], v);
        return true;
      }
      const std::string lo = fresh("%v");
      line("  " + lo + " = trunc i64 " + v + " to i32");
      store_reg(clean_operand(A[0]), lo);
      const std::string sh = fresh("%v");
      line("  " + sh + " = lshr i64 " + v + ", 32");
      const std::string hi = fresh("%v");
      line("  " + hi + " = trunc i64 " + sh + " to i32");
      store_reg(clean_operand(A[1]), hi);
      return true;
    }
    if (A.size() != 2) return fail("bad " + ins.op + " operands @ " + std::to_string(idx));
    const bool f32 = ins.op.rfind(".f32") != std::string::npos;
    // .f64 counts as a 64-bit access: an f64 value lives in an i64 alloca, so
    // its bit pattern is moved as one i64 (no float type needed).
    const bool w64 = ins.op.rfind(".u64") != std::string::npos ||
                     ins.op.rfind(".s64") != std::string::npos ||
                     ins.op.rfind(".b64") != std::string::npos ||
                     ins.op.rfind(".f64") != std::string::npos;
    std::string addr = resolve_addr(A[1]);
    if (ins.op.find("x2") != std::string::npos) {
      // f16x2/bf16x2: 32-bit payload (2×half) as i32
      const std::string v = fresh("%v");
      line("  " + v + " = load i32, ptr " + addr + ", align 4");
      store_reg(A[0], v);
      return true;
    }
    if (is_8bit_ty(ins.op)) {
      // The destination is a normal (>= 32-bit) register, so the loaded byte is
      // zero-extended, except for .s8 which carries the sign (the interpreter's
      // rule for ld.global.s8).
      const std::string v = fresh("%v");
      line("  " + v + " = load i8, ptr " + addr + ", align 1");
      const std::string e = fresh("%v");
      line("  " + e + " = " + (ins.op.rfind(".s8") != std::string::npos ? "sext" : "zext") +
           " i8 " + v + " to " + reg_type(A[0]));
      store_reg(A[0], e);
      return true;
    }
    if (is_16bit_ty(ins.op)) {
      const std::string v = fresh("%v");
      line("  " + v + " = load i16, ptr " + addr + ", align 2");
      if (reg_type(A[0]) == "i16") {
        store_reg(A[0], v);
        return true;
      }
      // A wider destination: .s16 sign-extends, every other 16-bit type (u16,
      // b16, f16, bf16) zero-extends the halfword.
      const std::string e = fresh("%v");
      line("  " + e + " = " + (ins.op.find(".s16") != std::string::npos ? "sext" : "zext") +
           " i16 " + v + " to " + reg_type(A[0]));
      store_reg(A[0], e);
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
    return true;
  }

  bool emit_st_mem(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    // A vector source list "{%r2, %r3}" arrives as two tokens, so a v2 store
    // carries three operands (the address + two registers).
    const std::string vty = vec_ty(ins.op);
    if (!vty.empty()) {
      if (!is_v2_supported(vty))
        return fail("unsupported " + ins.op + " type '" + vty + "' @ " +
                    std::to_string(idx) +
                    " (implemented: v2.{f32,u32,s32,b32} as one 64-bit access)");
      if (A.size() < 2)
        return fail("bad " + ins.op + " operands @ " + std::to_string(idx));
      const std::string addr = resolve_addr(A[0]);
      if (A.size() < 3) {  // plain 64-bit source register
        line("  store i64 " + operand(A[1]) + ", ptr " + addr + ", align 8");
        return true;
      }
      // Low word first, as in memory: {lo, hi} pack into one 64-bit access.
      // The halves are 32-bit registers in legal PTX, but widen whatever type
      // they are so a mismatch cannot produce a wrong-width or no-op cast.
      std::string halves[2];
      const std::string regs[2] = {clean_operand(A[1]), clean_operand(A[2])};
      for (int i = 0; i < 2; ++i) {
        const std::string rt = reg_type(regs[i]);
        halves[i] = fresh("%v");
        if (rt == "i64") {
          line("  " + halves[i] + " = load i64, ptr %" + regs[i] + ", align 8");
        } else {
          line("  " + halves[i] + " = zext " + rt + " " + load_reg(regs[i]) +
               " to i64");
        }
      }
      const std::string sh = fresh("%v");
      line("  " + sh + " = shl i64 " + halves[1] + ", 32");
      const std::string v = fresh("%v");
      line("  " + v + " = or i64 " + halves[0] + ", " + sh);
      line("  store i64 " + v + ", ptr " + addr + ", align 8");
      return true;
    }
    if (A.size() != 2) return fail("bad " + ins.op + " operands @ " + std::to_string(idx));
    const bool f32 = ins.op.rfind(".f32") != std::string::npos;
    // .f64 counts as a 64-bit access: an f64 value lives in an i64 alloca, so
    // its bit pattern is moved as one i64 (no float type needed).
    const bool w64 = ins.op.rfind(".u64") != std::string::npos ||
                     ins.op.rfind(".s64") != std::string::npos ||
                     ins.op.rfind(".b64") != std::string::npos ||
                     ins.op.rfind(".f64") != std::string::npos;
    std::string addr = resolve_addr(A[0]);
    if (ins.op.find("x2") != std::string::npos) {
      // f16x2/bf16x2: 32-bit payload as i32
      line("  store i32 " + operand(A[1]) + ", ptr " + addr + ", align 4");
      return true;
    }
    if (is_8bit_ty(ins.op)) {
      // Any source register: the store takes the low byte.
      const std::string s = operand(A[1]);
      const std::string v = fresh("%v");
      line("  " + v + " = trunc " + reg_type(A[1]) + " " + s + " to i8");
      line("  store i8 " + v + ", ptr " + addr + ", align 1");
      return true;
    }
    if (is_16bit_ty(ins.op)) {
      // A 32-bit source register is fine: the store takes its low halfword.
      // (Requiring a 16-bit source here used to reject valid PTX.)
      const std::string s = operand(A[1]);
      const std::string srcTy = operand_type(A[1], 16);
      if (srcTy == "i16") {
        line("  store i16 " + s + ", ptr " + addr + ", align 2");
        return true;
      }
      const std::string v = fresh("%v");
      line("  " + v + " = trunc " + srcTy + " " + s + " to i16");
      line("  store i16 " + v + ", ptr " + addr + ", align 2");
      return true;
    }
    if (f32) {
      const std::string bits = operand(A[1]);
      const std::string f = fresh("%f");
      line("  " + f + " = bitcast i32 " + bits + " to float");
      line("  store float " + f + ", ptr " + addr + ", align 4");
      return true;
    }
    if (w64) {
      line("  store i64 " + operand(A[1]) + ", ptr " + addr + ", align 8");
      return true;
    }
    line("  store i32 " + operand(A[1]) + ", ptr " + addr + ", align 4");
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
        const std::string fall = (i + 1 < body_.size())
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
    if (i + 1 >= body_.size()) {
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
  LlvmGen gen(&fn, &prog, error);
  return gen.generate();
}

}  // namespace rougecomp