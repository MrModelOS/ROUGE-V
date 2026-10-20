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

// Address space named by the instruction itself ("ld.global", "st.shared").
// The opcode outranks the register mark: a register that once held a shared
// address may later hold a global one (gemm_tile reuses %rd3 for the C tile
// store), and only the instruction knows which access this one is.
std::string op_space(const std::string& op) {
  if (op.find(".shared") != std::string::npos) return "shared";
  if (op.find(".global") != std::string::npos) return "global";
  return "";
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
          std::string* error, std::string target = "x86_64-pc-linux-gnu")
      : fn_(fn), prog_(prog), target_(std::move(target)), error_(error) {}

  std::string generate() {
    normalize_body();
    if (body_.empty()) {
      fail("kernel has an empty body");
      return "";
    }
    build_block_names();
    if (!scan_warp_collectives()) return "";
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
  std::string target_ = "x86_64-pc-linux-gnu";
  std::set<std::string> shared_addr_regs_;  // written by cvta.to.shared
  std::set<std::string> local_addr_regs_;  // .local depot addresses (generic ptrs)
  std::string smem_sym_;    // GPU block scratchpad global
  std::string smem_decl_;  // its module-level declaration, emitted with the header
  // Pointer type per SSA value, so memory operations can print "ptr addrspace(N)"
  // for GPU targets and plain "ptr" for CPU targets.
  std::map<std::string, std::string> ptr_ty_;
  // Type to print in front of a pointer operand.
  std::string pt(const std::string& addr) const {
    const auto it = ptr_ty_.find(addr);
    return it == ptr_ty_.end() ? std::string("ptr") : it->second;
  }

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
  // Значение предиката текущей инструкции (i1) — пусто, если инструкция не
  // предицированная. Служебные случаи, где регистр пишется ВСЕГДА (shfl с
  // предикатом назначения: там предикат значит «источник в диапазоне»), —
  // через always_writes_.
  std::string cur_pred_;
  bool always_writes_ = false;
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

  // A token names a register when it is declared in .reg (covers %SP/%SPL,
  // which end in no digit) or matches the %r0/%rd1/%p2 pattern (covers
  // undeclared uses, which default to 32 bits). Symbols (.shared/.global/
  // .local names, special registers) live in their own tables and never
  // match, so this cannot misclassify them.
  bool is_reg(const std::string& tok) const {
    if (fn_->regBits.find(tok) != fn_->regBits.end()) return true;
    return is_register(tok);
  }

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
    line("  " + v + " = load " + ty + ", " + pt("%" + reg) + " %" + reg + ", align " + align_of(ty));
    return v;
  }
  // Widen a value to i64 (zext). Literals and i64 values pass through; this
  // keeps the prmt lowering readable.
  std::string zext_to_i64(const std::string& v, const std::string& ty) {
    if (ty == "i64") return v;
    const std::string w = fresh("%zx");
    line("  " + w + " = zext " + ty + " " + v + " to i64");
    return w;
  }
  void store_reg(const std::string& reg, const std::string& ssa) {
    const std::string ty = reg_type(reg);
    // PTX ISA 9.7.3: предицированная инструкция при ЛОЖНОМ предикате не
    // выполняется, и регистр-назначение НЕ пишется — там остаётся старое
    // значение. Раньше запись была безусловной, и карта считала иначе
    // (нашлось дифференциальным фазером).
    if (!cur_pred_.empty() && !always_writes_) {
      const std::string old = load_reg(reg);
      const std::string sel = fresh("%psel");
      line("  " + sel + " = select i1 " + cur_pred_ + ", " + ty + " " + ssa +
           ", " + ty + " " + old);
      line("  store " + ty + " " + sel + ", " + pt("%" + reg) + " %" + reg +
           ", align " + align_of(ty));
      return;
    }
    line("  store " + ty + " " + ssa + ", " + pt("%" + reg) + " %" + reg + ", align " + align_of(ty));
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
    return is_reg(tok) ? reg_type(tok) : int_ty(bits);
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
        op.rfind("bfi.", 0) == 0 || op.rfind("bfe.", 0) == 0 ||
        op.rfind("prmt.", 0) == 0 || op.rfind("dp4a.", 0) == 0 ||
        op.rfind("dp2a.", 0) == 0)
      return "unary";
    return "";
  }

  // CUDA special register read. On the host these come out of the launch
  // descriptor; on a GPU they are hardware registers reached through NVVM.
  // Either way the result is the thread's index and nothing else depends on
  // where it came from.
  std::string load_spec(const std::string& name) {
    if (is_nvptx()) {
      const std::string v = fresh("%v");
      line("  " + v + " = call i32 @" + nvvm_sreg(name) + "()");
      return v;
    }
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
    line("target triple = \"" + target_ + "\"");
    emit_global_decls();
    emit_shared_metadata();
    // The scratchpad has to be declared before the body that addresses it, and
    // the body is what first asks for its size, so size it here.
    if (is_nvptx() && !fn_->sharedVars.empty()) {
      smem_global();
      line(smem_decl_);
      // One extern window per dynamic (extern, unsized) variable: the same
      // shape CUDA clang emits for `extern __shared__`, storage provided by
      // the launch's dynamic-shared allocation.
      for (const auto& sv : fn_->sharedVars) {
        if (!sv.dynamic) continue;
        line("@" + dsmem_sym(sv.name) +
             " = external addrspace(3) global [0 x i8], align " +
             std::to_string(std::max(sv.align, 1)));
      }
    }
  }

  // True for every barrier mnemonic we lower: bar.sync, its new spelling
  // barrier.sync, and the warp-scoped bar.warp.sync (implied by the
  // block-wide barrier: stronger, still correct). Split-phase arrive/red
  // forms (bar.arrive, barrier.red, mbarrier.*) are NOT matched: they do
  // not block and lowering them to a full barrier would be wrong.
  static bool is_barrier_op(const std::string& op) {
    return op == "bar.sync" || op == "bar.warp.sync" || op == "barrier.sync";
  }

  void emit_signature() {
    // Module-level declare for the block barrier runtime helper (bar.sync).
    for (const auto& ins : body_)
      if (is_barrier_op(ins.op)) use_sync_ = true;
    // On a real GPU the barrier is a hardware instruction reached through
    // NVVM, not a call into a host pthread barrier.
    if (use_sync_ && !is_nvptx()) line("declare void @__rouge_syncthreads(ptr)");
    scan_intrinsics();
    if (gpu_kernel()) emit_gpu_intrinsics();
    for (const auto& d : intrinsics_) line(d);
    // Calling convention 71 is LLVM's NVVM_KERNEL. It is what makes clang
    // emit ".visible .entry" (a launchable kernel) instead of ".visible .func"
    // (an ordinary device function the driver will not accept).
    std::string sig = "define " + std::string(gpu_kernel() ? "cc 71 void" : "void") +
                      " @" + fn_->name + "(";
    for (size_t i = 0; i < fn_->params.size(); ++i) {
      if (i) sig += ", ";
      sig += param_llvm_type(fn_->params[i]) + " %arg" + std::to_string(i);
    }
    // The launch descriptor carries the grid/block shape and the block
    // scratchpad. On a GPU both come from the hardware and the shared window,
    // so the descriptor has no place in the signature.
    if (!gpu_kernel()) {
      if (!fn_->params.empty()) sig += ", ";
      sig += "ptr %launch";
    }
    sig += ") {";
    line(sig);
  }

  // True when the target is a real GPU and the kernel has to be built the way
  // that GPU launches one, rather than as a host function.
  //
  // NVPTX only, for now. AMDGPU needs the same treatment (workitem ids from
  // llvm.amdgcn.*, the barrier from s_barrier, and its own kernel calling
  // convention), and until that exists AMD keeps the host calling convention
  // with the launch descriptor: it must not lose the parameter it reads from.
  bool gpu_kernel() const { return is_nvptx(); }

  // The NVVM intrinsics stand in for the launch descriptor: the special
  // registers come from the hardware, and the block scratchpad is a module
  // global in the shared address space.
  void emit_gpu_intrinsics() {
    for (const auto& ins : body_) {
      if (ins.args.size() < 2) continue;
      if (spec_offset(ins.args[1]) < 0 &&
          sreg_llvm_ty(ins.args[1]).empty())
        continue;
      if (!ins.args[1].empty() && ins.args[1][0] == '%') continue;
      nvvm_sreg(ins.args[1]);
    }
    if (use_sync_) {
      intrinsics_.insert("declare void @llvm.nvvm.barrier0()");
      intrinsics_.insert("declare void @llvm.nvvm.barrier0.aligned(i32, i32)");
    }
    // Real warp shuffles need their declares up front with the signature: the
    // body that calls them is emitted afterwards. Same predicate as the call
    // site below, so the two cannot disagree.
    for (const auto& ins : body_) {
      if (ins.op.rfind("shfl", 0) != 0) continue;
      std::string kind;
      bool sync = false;
      if (!nvptx_shfl(ins, &kind, &sync)) continue;
      // Always the .sync form: the legacy (mask-less) NVVM intrinsic crashes
      // this LLVM backend, so legacy PTX passes member mask -1 (full warp).
      intrinsics_.insert("declare i32 @llvm.nvvm.shfl.sync." + kind +
                         ".i32(i32, i32, i32, i32)");
      // A predicated destination ("d|p") needs the lane id up front too:
      // the predicate is computed from laneid + offset in the body below.
      if (ins.args[0].find('|') != std::string::npos &&
          (kind == "down" || kind == "up"))
        nvvm_sreg("laneid");
    }
    // Real warp votes need their declares up front too. Backend-verified
    // signature: i1 @llvm.nvvm.vote.<kind>.sync(i32 mask, i1 pred).
    for (const auto& ins : body_) {
      if (ins.op.rfind("vote", 0) != 0) continue;
      std::string kind;
      bool sync = false;
      if (!nvptx_vote(ins, &kind, &sync)) continue;
      intrinsics_.insert("declare i1 @llvm.nvvm.vote." + kind +
                         ".sync(i32, i1)");
    }
    // activemask.b32 reads the executing mask straight from the hardware.
    for (const auto& ins : body_) {
      if (ins.op.rfind("activemask", 0) != 0) continue;
      if (ins.args.size() != 1) continue;
      intrinsics_.insert("declare i32 @llvm.nvvm.activemask()");
    }
  }

  // Emit (once) a read of the named special register and return the intrinsic.
  // Most sregs are i32; clock64 is i64 (sreg_llvm_ty knows).
  std::string nvvm_sreg(const std::string& name) {
    const std::string ty =
        sreg_llvm_ty(name).empty() ? "i32" : sreg_llvm_ty(name);
    const std::string fn = "llvm.nvvm.read.ptx.sreg." + name;
    intrinsics_.insert("declare " + ty + " @" + fn + "()");
    return fn;
  }

  // Cycle/id special registers outside the launch descriptor: the LLVM type
  // they read as, or "" when the name is not one. NVPTX serves them from real
  // hardware sregs; the host has no equivalent (notably %clock64: CPU wall
  // time is not GPU cycles, so a host fallback would silently lie about units).
  static std::string sreg_llvm_ty(const std::string& name) {
    if (name == "clock64") return "i64";
    if (name == "clock" || name == "warpid" || name == "nwarpid" ||
        name == "smid" || name == "nsmid" || name == "gridid" ||
        name == "laneid")
      return "i32";
    return "";
  }

  // NVVM vote kind named by the opcode, or "" when this vote must stay on
  // the scalar path. `sync` reports whether the opcode already carries the
  // .sync form (with an explicit member mask as its last operand).
  std::string vote_nvvm_kind(const std::string& op, bool* sync) {
    if (sync) *sync = op.find(".sync.") != std::string::npos;
    if (op.find(".any.") != std::string::npos) return "any";
    if (op.find(".all.") != std::string::npos) return "all";
    if (op.find(".uni.") != std::string::npos) return "uni";
    return "";
  }

  // True when this vote can go through a real NVVM ballot: NVPTX target,
  // known kind, .pred (i1) destination, and the operand count of its form
  // (2 legacy, 3 .sync). Backend-verified signature:
  //   i1 @llvm.nvvm.vote.<kind>.sync(i32 mask, i1 pred)
  // Legacy (mask-less) PTX passes member mask -1, i.e. the full warp.
  bool nvptx_vote(const rouge::PtxInstruction& ins, std::string* kind,
                  bool* sync) {
    if (!is_nvptx()) return false;
    const std::string k = vote_nvvm_kind(ins.op, sync);
    if (k.empty()) return false;
    const size_t want = *sync ? 3 : 2;
    if (ins.args.size() != want) return false;
    if (reg_type(ins.args[0]) != "i1") return false;
    *kind = k;
    return true;
  }
  // NVVM warp-shuffle kind named by the opcode, or "" when this shfl must stay
  // on the scalar path. `sync` reports whether the opcode already carries the
  // .sync form (with an explicit member mask as its last operand).
  std::string shfl_nvvm_kind(const std::string& op, bool* sync) {
    if (sync) *sync = op.find(".sync.") != std::string::npos;
    if (op.find(".idx.") != std::string::npos) return "idx";
    if (op.find(".up.") != std::string::npos) return "up";
    if (op.find(".down.") != std::string::npos) return "down";
    if (op.find(".bfly.") != std::string::npos) return "bfly";
    return "";
  }

  // True when this shfl instruction can go through a real NVVM shuffle:
  // NVPTX target, known kind, i32 destination (with optional SET-predicate
  // "d|p" per PTX ISA 9.7.10.6) and source, and the operand count of its
  // form (4 legacy, 5 .sync).
  bool nvptx_shfl(const rouge::PtxInstruction& ins, std::string* kind, bool* sync) {
    if (!is_nvptx()) return false;
    const std::string k = shfl_nvvm_kind(ins.op, sync);
    if (k.empty()) return false;
    const size_t want = *sync ? 5 : 4;
    if (ins.args.size() != want) return false;
    // Strip the optional destination SET-predicate before checking the type.
    std::string dst = ins.args[0];
    const size_t bar = dst.find('|');
    if (bar != std::string::npos) dst = dst.substr(0, bar);
    if (reg_type(dst) != "i32") return false;
    if (is_reg(ins.args[1]) && reg_type(ins.args[1]) != "i32") return false;
    *kind = k;
    return true;
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
      // fma.rn.f32 and the widened f16/bf16 fma both reach llvm.fma.f32.
      if (op.rfind("fma.", 0) == 0 &&
          (op.find(".f32") != std::string::npos ||
           op.find(".f16") != std::string::npos ||
           op.find(".bf16") != std::string::npos)) {
        intrinsics_.insert(
            "declare float @llvm.fma.f32(float, float, float)");
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
    const auto add_reg = [&](const std::string& tok) {
      // Hardware special registers (%clock64, %warpid, ...) are read from
      // intrinsics, never from an alloca: allocating one would silently feed
      // garbage into the kernel (clock.ptx caught this).
      if (!sreg_llvm_ty(tok).empty()) return;
      if (is_reg(tok) && !is_label(tok)) used.insert(tok);
      // Predicated destinations arrive as "r10|p1" — allocate the guard too.
      const size_t bar = tok.find('|');
      if (bar != std::string::npos && bar + 1 < tok.size()) {
        std::string g = tok.substr(bar + 1);
        if (!g.empty() && g[0] == '%') g = g.substr(1);
        if (is_reg(g) && !is_label(g)) used.insert(g);
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
      // AMDGPU rejects a generic alloca outright ("alloca on amdgpu must be in
      // addrspace(5)"): it has no stack and keeps private objects in the shared
      // address space. LLVM 22 wants the address space AFTER the alignment,
      // i.e. "alloca T, align N, addrspace(5)".
      const std::string as = is_amdgpu() ? ", addrspace(5)" : "";
      const std::string selfTy = is_amdgpu() ? "ptr addrspace(5)" : "ptr";
      line("  %" + r + " = alloca " + ty + ", align " + align_of(ty) + as);
      ptr_ty_["%" + r] = selfTy;  // load_reg/store_reg must agree
      // A .pred register the kernel never wrote reads as poison otherwise; the
      // interpreter treats an unset predicate as false, so seed it that way.
      if (ty == "i1")
        line("  store i1 false, " + selfTy + " %" + r + ", align 1");
    }
    // Per-thread .local spill depots (nvcc "__local_depotN" arrays): one
    // generic alloca each, addressed with ptrtoint like any other stack slot.
    for (const auto& lv : fn_->localVars) {
      const std::string as = is_amdgpu() ? ", addrspace(5)" : "";
      const std::string selfTy = is_amdgpu() ? "ptr addrspace(5)" : "ptr";
      line("  %" + lv.name + " = alloca [" + std::to_string(lv.bytes) +
           " x i8], align " + std::to_string(lv.align > 0 ? lv.align : 1) +
           as);
      ptr_ty_["%" + lv.name] = selfTy;
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

    // Предикат текущей инструкции: грузим один раз, дальше store_reg()
    // сам выбирает «писать или оставить старое значение».
    cur_pred_.clear();
    always_writes_ = false;
    if (!ins.pred.empty()) {
      const std::string pname =
          ins.pred[0] == '!' ? ins.pred.substr(1) : ins.pred;
      if (is_reg(pname)) {
        std::string pv = load_reg(pname);
        if (ins.pred[0] == '!') {
          const std::string nv = fresh("%np");
          line("  " + nv + " = xor i1 " + pv + ", true");
          pv = nv;
        }
        cur_pred_ = pv;
        // shfl: предикат в НАЗНАЧЕНИИ — это «источник в диапазоне», а не
        // «выполнить или нет»: значение пишется всегда.
        if (op.rfind("shfl", 0) == 0) always_writes_ = true;
      }
    }

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
      // Cycle/id special registers (%clock64, %warpid, ...): NVPTX reads the
      // real hardware counter; the host has no equivalent and fails loudly
      // instead of wiring an uninitialised alloca (clock.ptx caught this).
      const std::string sregTy = sreg_llvm_ty(A[1]);
      if (!sregTy.empty()) {
        if (!is_nvptx())
          return fail("unsupported " + op + " of %" + A[1] + " on host @ " +
                      std::to_string(idx) +
                      " (no host equivalent of this hardware register)");
        const std::string v = fresh("%v");
        line("  " + v + " = call " + sregTy + " @" + nvvm_sreg(A[1]) + "()");
        store_reg(A[0], fit_reg(A[0], v, sregTy));
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
        const std::string base = shared_base_for(A[1]);
        const std::string addr = fresh("%sa");
        line("  " + addr + " = add i64 " + base + ", " +
             std::to_string(symOff + symConst));
        shared_addr_regs_.insert(A[0]);  // result is a shared-memory address
        store_reg(A[0], fit_reg(A[0], addr, "i64"));
        return true;
      }
      // "mov.b64 %SPL, __local_depot0" materialises the thread depot address
      // (the form nvcc emits for spilled stack frames).
      if (find_local(A[1])) {
        store_reg(A[0], fit_reg(A[0], local_addr(A[1]), "i64"));
        local_addr_regs_.insert(A[0]);  // stays a generic pointer
        return true;
      }
      // Width mismatch (e.g. mov.b16 %h, %r32): fit the bit pattern,
      // growing with zext and shrinking with trunc, like the interpreter's
      // width_mask (an unconditional trunc miscompiles i32 -> i64 moves).
      const std::string dstTy = reg_type(A[0]);
      std::string val = operand(A[1]);  // register or numeric immediate
      if (is_reg(A[1]) && reg_type(A[1]) != dstTy)
        val = fit_reg(A[0], val, reg_type(A[1]));
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
      const std::string base = shared_base_for(A[1]);
      const std::string addr = fresh("%sa");
      line("  " + addr + " = add i64 " + base + ", " + std::to_string(off + c));
      shared_addr_regs_.insert(A[0]);  // result is a shared-memory address
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
    if (op.rfind("cvta.local.", 0) == 0) {
      // cvta.local.u64 %SP, %SPL: the source register already holds the depot
      // address (materialised by mov from the .local symbol above); the
      // conversion is an identity copy, width-fitted like the interpreter.
      if (A.size() != 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      store_reg(A[0], fit_reg(A[0], operand(A[1]), reg_type(A[1])));
      local_addr_regs_.insert(A[0]);  // stays a generic pointer
      return true;
    }
    if (op.rfind("cvta.", 0) == 0) {
      // Plain cvta.space without .to (the form clang's CUDA frontend emits,
      // e.g. cvta.shared.u64 %rd, %r): a register-to-register address-space
      // annotation. Addresses are plain i64 everywhere here, so this is an
      // identity copy, width-fitted like the interpreter — but the window
      // provenance (shared/local marks) must flow through, or generic
      // accesses via the result resolve to the wrong address space.
      if (A.size() != 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      // A symbol source (cvta.shared.u64 %rd, sym) takes the shared window
      // path with its laid-out offset, like cvta.to.shared.
      int symOff = 0, symConst = 0;
      if (shared_symbol(A[1], &symOff, &symConst)) {
        const std::string base = shared_base_for(A[1]);
        const std::string addr = fresh("%sa");
        line("  " + addr + " = add i64 " + base + ", " +
             std::to_string(symOff + symConst));
        shared_addr_regs_.insert(A[0]);
        store_reg(A[0], fit_reg(A[0], addr, "i64"));
        return true;
      }
      const std::string srcTy =
          is_reg(A[1]) ? reg_type(A[1]) : std::string("i64");
      store_reg(A[0], fit_reg(A[0], operand(A[1]), srcTy));
      if (is_reg(A[1]) && is_shared_addr_reg(A[1]))
        shared_addr_regs_.insert(A[0]);
      if (is_reg(A[1]) && is_local_addr_reg(A[1]))
        local_addr_regs_.insert(A[0]);
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

    if (op.rfind("ld.", 0) == 0 && op.rfind("ld.param.", 0) != 0) {
      return emit_ld_mem(ins, idx);
    }
    if (op.rfind("st.", 0) == 0) {
      return emit_st_mem(ins, idx);
    }

    // ---- warp shuffles: shfl.sync.<kind>.b32 (and legacy shfl.<kind>.b32) ----
    // On a real GPU a shuffle exchanges lanes, so the scalar identity fallback
    // below would silently compute the wrong answer there. NVPTX therefore
    // goes through the NVVM intrinsic (always the .sync form: the legacy
    // mask-less intrinsic crashes this LLVM backend, so legacy PTX passes
    // member mask -1, i.e. the full warp). Host targets keep the scalar
    // fallback: shfl is identity before vectorization, MLIR will vectorize.
    if (op.rfind("shfl", 0) == 0) {
      std::string kind;
      bool sync = false;
      // The destination may carry a SET-predicate: "shfl.sync.down.b32 %r10|%p1, ...".
      // Per PTX ISA 9.7.10.6 p is SET, not a guard: True when the computed
      // source lane is in [0,31], False otherwise. The value is always written.
      std::string dst = A[0];
      std::string setPred;
      {
        const size_t bar = dst.find('|');
        if (bar != std::string::npos) {
          setPred = dst.substr(bar + 1);
          dst = dst.substr(0, bar);
          if (!setPred.empty() && setPred[0] == '%') setPred = setPred.substr(1);
          if (setPred.empty() || !is_reg(setPred))
            return fail("bad shfl set predicate '" + A[0] + "' @ " +
                        std::to_string(idx));
        }
      }
      if (nvptx_shfl(ins, &kind, &sync)) {
        // A = [dst, src, b, c] legacy or [dst, src, b, c, mask] .sync.
        // Immediates arrive in PTX spelling ("0x1f"), which LLVM IR does not
        // accept as an i32 literal: spell them decimal.
        const auto imm = [](const std::string& t) -> std::string {
          if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X'))
            return std::to_string(
                static_cast<long long>(std::strtoll(t.c_str(), nullptr, 16)));
          return t;
        };
        const std::string src = is_reg(A[1]) ? operand(A[1]) : imm(A[1]);
        const std::string b = is_reg(A[2]) ? operand(A[2]) : imm(A[2]);
        const std::string c = is_reg(A[3]) ? operand(A[3]) : imm(A[3]);
        const std::string mask =
            sync ? (is_reg(A[4]) ? operand(A[4]) : imm(A[4])) : "-1";
        const std::string v = fresh("%sh");
        line("  " + v + " = call i32 @llvm.nvvm.shfl.sync." + kind +
             ".i32(i32 " + mask + ", i32 " + src + ", i32 " + b + ", i32 " + c +
             ")");
        store_reg(dst, v);
        // SET-predicate per PTX ISA 9.7.10.6, hardware-verified on sm_75:
        // the clamp operand (c) is a RANGE BOUND, not a fallback lane.
        // .down: valid iff lane+off <= clamp; .up: valid iff lane>=off and
        // lane-off >= clamp. .bfly/.idx are always in range. Out-of-range
        // lanes keep their own value with p=False.
        if (!setPred.empty()) {
          if (kind == "bfly" || kind == "idx") {
            store_reg(setPred, "true");
          } else {
            const std::string lane = fresh("%v");
            line("  " + lane + " = call i32 @" + nvvm_sreg("laneid") + "()");
            const std::string off5 = fresh("%shoff");
            line("  " + off5 + " = and i32 " + b + ", 31");
            const std::string cl5 = fresh("%shcl");
            line("  " + cl5 + " = and i32 " + c + ", 31");
            const std::string pred = fresh("%shp");
            if (kind == "down") {
              const std::string sum = fresh("%shsum");
              line("  " + sum + " = add i32 " + lane + ", " + off5);
              line("  " + pred + " = icmp ule i32 " + sum + ", " + cl5);
            } else {  // up
              const std::string ok = fresh("%shok");
              line("  " + ok + " = icmp uge i32 " + lane + ", " + off5);
              const std::string src = fresh("%shsrc");
              line("  " + src + " = sub i32 " + lane + ", " + off5);
              const std::string inb = fresh("%shin");
              line("  " + inb + " = icmp uge i32 " + src + ", " + cl5);
              line("  " + pred + " = and i1 " + ok + ", " + inb);
            }
            store_reg(setPred, pred);
          }
        }
        return true;
      }
      if (A.size() < 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      line("  ; scalar fallback: shfl is identity before vectorization, MLIR will vectorize");
      // A[0]=dst, A[1]=src (value to shuffle); A[2]=lane/offset, A[3]=clamp.
      // dst/setPred already split above; the value is always written and the
      // predicate is SET per PTX ISA 9.7.10.6.
      // Width mismatch (e.g. an immediate): fit the bit pattern, growing
      // with zext and shrinking with trunc, like the interpreter's width_mask.
      const std::string dstTy = reg_type(dst);
      std::string srcVal = operand(A[1]);
      if (is_reg(A[1]) && reg_type(A[1]) != dstTy)
        srcVal = fit_reg(dst, srcVal, reg_type(A[1]));
      // Always write the value; SET the predicate per PTX ISA 9.7.10.6.
      store_reg(dst, srcVal);
      if (!setPred.empty()) {
        bool syncDummy = false;
        const std::string k2 = shfl_nvvm_kind(op, &syncDummy);
        if (k2 == "bfly" || k2 == "idx") {
          store_reg(setPred, "true");
        } else if (k2 == "down" || k2 == "up") {
          // Host scalar path: one lane at a time, lane = tid.x mod 32.
          // Bound model, hardware-verified: down valid iff lane+off<=clamp,
          // up valid iff lane>=off and lane-off>=clamp (clamp defaults to
          // 31 when the form carries none).
          const auto dec = [](const std::string& t) -> std::string {
            if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X'))
              return std::to_string(
                  static_cast<long long>(std::strtoll(t.c_str(), nullptr, 16)));
            return t;
          };
          const std::string tid = load_spec("tid.x");
          const std::string laneMod = fresh("%shlm");
          line("  " + laneMod + " = urem i32 " + tid + ", 32");
          std::string offRaw =
              is_reg(A[2]) ? operand(A[2]) : normalize_literal(A[2]);
          offRaw = dec(offRaw);
          const std::string off5 = fresh("%shoff");
          line("  " + off5 + " = and i32 " + offRaw + ", 31");
          std::string clRaw("31");
          if (A.size() >= 4)
            clRaw = is_reg(A[3]) ? operand(A[3])
                                      : dec(normalize_literal(A[3]));
          const std::string cl5 = fresh("%shcl");
          line("  " + cl5 + " = and i32 " + clRaw + ", 31");
          const std::string pred = fresh("%shp");
          if (k2 == "down") {
            const std::string sum = fresh("%shsum");
            line("  " + sum + " = add i32 " + laneMod + ", " + off5);
            line("  " + pred + " = icmp ule i32 " + sum + ", " + cl5);
          } else {  // up
            const std::string ok = fresh("%shok");
            line("  " + ok + " = icmp uge i32 " + laneMod + ", " + off5);
            const std::string src = fresh("%shsrc");
            line("  " + src + " = sub i32 " + laneMod + ", " + off5);
            const std::string inb = fresh("%shin");
            line("  " + inb + " = icmp uge i32 " + src + ", " + cl5);
            line("  " + pred + " = and i1 " + ok + ", " + inb);
          }
          store_reg(setPred, pred);
        } else {
          store_reg(setPred, "true");
        }
      }
      return true;
    }

    // ---- vote.{any,all,uni}.pred and vote.sync.* ----
    if (op.rfind("vote", 0) == 0) {
      if (A.size() < 2) return fail("bad " + op + " operands @ " + std::to_string(idx));
      std::string vkind;
      bool vsync = false;
      if (nvptx_vote(ins, &vkind, &vsync)) {
        // A = [dst.pred, src.pred, (mask)]. The source may be negated
        // ("!%p1"); the intrinsic takes an i1 either way.
        std::string srcPred = A[1];
        bool neg = false;
        if (!srcPred.empty() && srcPred[0] == '!') {
          neg = true;
          srcPred = srcPred.substr(1);
        }
        std::string pv = load_reg(srcPred);
        if (neg) {
          const std::string nv = fresh("%n");
          line("  " + nv + " = xor i1 " + pv + ", true");
          pv = nv;
        }
        // Immediates arrive in PTX spelling ("0xffffffff"), which LLVM IR
        // does not accept as an i32 literal: spell them decimal.
        const auto decimm = [](const std::string& t) -> std::string {
          if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X'))
            return std::to_string(
                static_cast<long long>(std::strtoll(t.c_str(), nullptr, 16)));
          return t;
        };
        const std::string mask =
            vsync ? (is_reg(A[2]) ? operand(A[2]) : decimm(A[2])) : "-1";
        const std::string v = fresh("%vt");
        line("  " + v + " = call i1 @llvm.nvvm.vote." + vkind + ".sync(i32 " +
             mask + ", i1 " + pv + ")");
        store_reg(A[0], v);
        return true;
      }
      // A warp collective has no meaning one lane at a time: the scalar
      // fallback is host-only. On a GPU it would silently answer wrong, so
      // refuse instead of emitting it.
      if (is_nvptx())
        return fail("unsupported " + op + " on NVPTX @ " + std::to_string(idx) +
                    " (warp-collective lowering not implemented yet)");
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
      // Backend-verified: i32 @llvm.nvvm.activemask() lowers to activemask.b32.
      if (is_nvptx()) {
        if (A.size() != 1)
          return fail("unsupported " + op + " form on NVPTX @ " +
                      std::to_string(idx) + " (want activemask.b32 %d)");
        const std::string v = fresh("%am");
        line("  " + v + " = call i32 @llvm.nvvm.activemask()");
        store_reg(A[0], fit_reg(A[0], v, "i32"));
        return true;
      }
      line("  ; scalar fallback: activemask -> all lanes active (0xffffffff)");
      // b32 destination is i32; -1 == 0xffffffff
      store_reg(A[0], "-1");
      return true;
    }

    if (is_barrier_op(op)) {
      // Block-wide barrier. On the host this is __rouge_syncthreads (a
      // pthread barrier); on a GPU it is the hardware barrier instruction,
      // which needs no operand and no argument block.
      if (is_nvptx()) {
        line("  call void @llvm.nvvm.barrier0()");
        return true;
      }
      // bar.sync 0[, count] / barrier.sync 0 — ignore barrier id for now
      // (single barrier).
      line("  call void @__rouge_syncthreads(ptr %launch)");
      return true;
    }

    // Memory fences and system barriers: __threadfence() -> membar.gl,
    // also membar.sys / membar.cta / fence.*. The semantics requested are
    // release/acquire on the memory space, which on both targets is already
    // provided by the underlying memory model: on the host the following
    // atomic RMWs are host atomics, and on the GPU the hardware enforces
    // them. A fence must never be dropped silently by accident, so it is an
    // explicit seq_cst fence (stronger than asked, never weaker).
    if (op.rfind("membar.", 0) == 0 || op.rfind("fence.", 0) == 0) {
      line("  fence seq_cst");
      return true;
    }

    if (op == "bra" || op == "bra.uni" || op == "ret" || op == "exit") return true;

    // Warp collectives: mma.sync exchanges fragments, ldmatrix/stmatrix only
    // row addresses, both through the per-block exchange region named by the
    // launch descriptor (see scan_warp_collectives). A GPU target is refused in
    // the pre-scan, where the missing region is diagnosed once.
    if (op.rfind("mma.", 0) == 0) return emit_mma(ins, idx);
    if (op.rfind("ldmatrix.", 0) == 0 || op.rfind("stmatrix.", 0) == 0)
      return emit_matrix_move(ins, idx);
    if (op.rfind("wmma.", 0) == 0 || op.rfind("movmatrix.", 0) == 0)
      return fail("warp-cooperative '" + op + "' @ " + std::to_string(idx) +
                  " is not lowered (wmma is a different matrix API; movmatrix"
                  " is a warp-wide permutation): refused loudly rather than"
                  " lowered as something else");

    return fail("unsupported PTX op '" + op + "' @ " + std::to_string(idx) +
                " (vectorizing lowering landing in MLIR stage 2)");
  }

  // ===================== warp collectives ============================
  //
  // The host runs one OS thread per lane, so a lane cannot read a neighbour's
  // register the way a warp can. Two mechanisms bridge that, both rooted in the
  // launch descriptor:
  //
  //  * mma.sync publishes its A/B fragments into a per-block exchange region
  //    (64 bytes per thread, named at descriptor offset 64) and reads the
  //    peers' words after a block barrier. The C/D fragments stay local: the
  //    tensor-core layout puts every element a lane's D fragment needs into
  //    exactly the peers whose A/B fragments it publishes, so only A and B
  //    cross lanes.
  //  * ldmatrix/stmatrix only need each lane's row START ADDRESS; the data
  //    itself lives in shared memory, so just the address is published.
  //
  // The block barrier is stronger than warp scope (it also waits for the
  // block's other warps): correct, not free. Both are refused on a GPU target,
  // where the hardware instructions do the work (see scan_warp_collectives).

  static constexpr int kExSlot = 64;     // bytes per thread
  static constexpr int kExAddrOff = 0;   // row-address field
  static constexpr int kExFragOff = 8;   // first fragment word
  static constexpr int kExMaxFrag = 14;  // (64 - 8) / 4

  static bool is_warp_collective_op(const std::string& op) {
    return op.rfind("mma.", 0) == 0 || op.rfind("wmma.", 0) == 0 ||
           op.rfind("ldmatrix.", 0) == 0 || op.rfind("stmatrix.", 0) == 0 ||
           op.rfind("movmatrix.", 0) == 0;
  }

  // Collect what the body needs before the signature is printed. On the host
  // the collectives publish into the exchange region, so the block barrier
  // declare is needed; on a GPU they are hardware instructions reached through
  // inline asm (probed working against clang: NVVM has no ldmatrix/mma
  // intrinsics in this form), needing nothing up front. wmma/movmatrix are
  // refused everywhere: a different matrix API and a warp permutation.
  bool scan_warp_collectives() {
    for (const auto& ins : body_) {
      if (!is_warp_collective_op(ins.op)) continue;
      if (ins.op.rfind("wmma.", 0) == 0 || ins.op.rfind("movmatrix.", 0) == 0)
        return fail("warp-cooperative '" + ins.op + "' is not lowered (wmma"
                    " is a different matrix API; movmatrix is a warp-wide"
                    " permutation): refused loudly rather than lowered as"
                    " something else");
      if (gpu_kernel()) continue;  // hardware instructions need no preamble
      use_sync_ = true;  // publishing and reading are separated by a barrier
      if (ins.op.rfind("mma.", 0) == 0) {
        // Half-precision decode/encode goes through the host runtime, so both
        // paths run the very same conversion code instead of two independent
        // transcriptions of the same table. Float accumulation needs the FMA
        // intrinsic declared up front with the signature (the body that calls
        // it is emitted afterwards).
        const std::string ty = ins.op;
        if (ty.find(".bf16") != std::string::npos)
          intrinsics_.insert("declare float @rouge_bf16_to_f32(i16)");
        if (ty.find(".f16") != std::string::npos) {
          intrinsics_.insert("declare float @rouge_f16_to_f32(i16)");
          intrinsics_.insert("declare i16 @rouge_f32_to_f16_rne(float)");
        }
        if (ty.find(".u8") == std::string::npos &&
            ty.find(".s8") == std::string::npos)
          intrinsics_.insert(
              "declare float @llvm.fma.f32(float, float, float)");
      }
    }
    return true;
  }

  // ---- exchange region --------------------------------------------------
  //
  // One context per collective: the exchange base, the linear tid and the warp
  // base are computed ONCE, then every publish/read addresses peers by offset
  // from them. The first version recomputed base+linear from the launch
  // descriptor on every word (134 base loads and 538 tid loads for a single
  // mma) — straight-line IR, so no LICM could ever clean it up. The cached
  // form emits 1 base load and 5 spec loads per collective.

  struct ExchangeCtx {
    std::string base;  // i64 exchange region base
    std::string warp;  // i32 warp base ((linear & ~31))
    std::string own;   // ptr this lane's slot
  };

  std::string load_exchange_base() {
    const std::string gep = fresh("%xg");
    line("  " + gep + " = getelementptr inbounds i8, ptr %launch, i64 64");
    const std::string v = fresh("%eb");
    line("  " + v + " = load i64, ptr " + gep + ", align 8");
    return v;
  }

  // Linear thread index in the block: tid.x + tid.y*ntid.x + tid.z*ntid.x*ntid.y
  // (the flattening the interpreter uses to find a warp's lanes).
  std::string linear_tid() {
    const std::string tx = load_spec("tid.x");
    const std::string ty = load_spec("tid.y");
    const std::string tz = load_spec("tid.z");
    const std::string nx = load_spec("ntid.x");
    const std::string ny = load_spec("ntid.y");
    const std::string a = i32op("mul", ty, nx);
    const std::string b = i32op("add", a, tx);
    const std::string xy = i32op("mul", nx, ny);
    const std::string c = i32op("mul", tz, xy);
    return i32op("add", b, c);
  }

  ExchangeCtx exchange_ctx() {
    ExchangeCtx x;
    x.base = load_exchange_base();
    const std::string lin = linear_tid();
    x.warp = i32op("and", lin, "-32");
    const std::string off = i32op("mul", lin, std::to_string(kExSlot));
    const std::string z = fresh("%oz");
    line("  " + z + " = zext i32 " + off + " to i64");
    const std::string a = fresh("%oa");
    line("  " + a + " = add i64 " + x.base + ", " + z);
    x.own = inttoptr_as(a, "ptr");
    return x;
  }

  // Slot of peer lane `lane' (warp + lane) as a `ptr` SSA.
  std::string peer_slot_ctx(const ExchangeCtx& x, const std::string& lane) {
    const std::string peer = i32op("add", x.warp, lane);
    const std::string off = i32op("mul", peer, std::to_string(kExSlot));
    const std::string z = fresh("%pz");
    line("  " + z + " = zext i32 " + off + " to i64");
    const std::string a = fresh("%pa");
    line("  " + a + " = add i64 " + x.base + ", " + z);
    return inttoptr_as(a, "ptr");
  }

  // Field at a byte offset inside a slot.
  std::string slot_field(const std::string& slotPtr, int byteOff, const char* tag) {
    const std::string f = fresh(tag);
    line("  " + f + " = getelementptr inbounds i8, ptr " + slotPtr + ", i64 " +
         std::to_string(byteOff));
    return f;
  }

  // Publish one fragment word of this lane.
  void publish_slot_ctx(const ExchangeCtx& x, const std::string& reg,
                        int index) {
    std::string v = operand(reg);
    if (is_reg(reg) && reg_type(reg) != "i32")
      v = fit_reg(reg, v, reg_type(reg));
    if (index < 0 || index >= kExMaxFrag) return;
    const std::string f =
        slot_field(x.own, kExFragOff + 4 * index, "%pf");
    line("  store i32 " + v + ", ptr " + f + ", align 4");
  }

  // Read word `word' published by peer lane `lane'.
  std::string read_frag_ctx(const ExchangeCtx& x, const std::string& lane,
                            int word) {
    const std::string f = slot_field(peer_slot_ctx(x, lane),
                                     kExFragOff + 4 * word, "%rf");
    const std::string v = fresh("%rv");
    line("  " + v + " = load i32, ptr " + f + ", align 4");
    return v;
  }

  // This lane's own published word `index' (read back from the slot).
  std::string own_slot_word_ctx(const ExchangeCtx& x, int index) {
    const std::string f =
        slot_field(x.own, kExFragOff + 4 * index, "%ow");
    const std::string v = fresh("%ov");
    line("  " + v + " = load i32, ptr " + f + ", align 4");
    return v;
  }

  // Publish / read the row address a lane computed (ldmatrix/stmatrix).
  void publish_row_addr_ctx(const ExchangeCtx& x, const std::string& addr) {
    const std::string f = slot_field(x.own, kExAddrOff, "%pa");
    line("  store i64 " + addr + ", ptr " + f + ", align 8");
  }
  std::string read_peer_row_ctx(const ExchangeCtx& x, const std::string& lane) {
    const std::string f =
        slot_field(peer_slot_ctx(x, lane), kExAddrOff, "%ra");
    const std::string v = fresh("%rw");
    line("  " + v + " = load i64, ptr " + f + ", align 8");
    return v;
  }

  // Lane index inside the warp. PTX numbers lanes by (tid.x % 32, tid.y % 32);
  // the block shapes the collectives support are 1-D, so tid.x % 32 is it.
  std::string lane_index() {
    const std::string t = load_spec("tid.x");
    return i32op("and", t, "31");
  }

  // Split operands into brace groups. Operands usually arrive split on ',',
  // so a group is a run of tokens between braces ("{%d0", "d1", ..., "d3}");
  // a single "{d0 d1 d2 d3}" token (whitespace-separated) is split the same
  // way, so both spellings give the same groups.
  static std::vector<std::vector<std::string>> brace_groups(
      const std::vector<std::string>& args) {
    std::vector<std::vector<std::string>> groups;
    std::vector<std::string> cur;
    bool inGroup = false;
    const auto flush = [&]() {
      if (!cur.empty()) {
        groups.push_back(cur);
        cur.clear();
      }
      inGroup = false;
    };
    for (const auto& a : args) {
      std::string t = a;
      const bool opens = t.find('{') != std::string::npos;
      const bool closes = t.find('}') != std::string::npos;
      if (opens && !cur.empty()) flush();
      if (opens) inGroup = true;
      // Split "{d0 d1}" into its registers; commas are already gone.
      std::string tok;
      for (size_t i = 0; i <= t.size(); ++i) {
        const char c = i < t.size() ? t[i] : ' ';
        if (c == ' ' || c == '\t' || c == ',') {
          if (!tok.empty()) {
            const std::string r = clean_operand(tok);
            if (!r.empty()) cur.push_back(r);
            tok.clear();
          }
        } else {
          tok += c;
        }
      }
      if (closes) flush();
    }
    flush();
    return groups;
  }

  // ---- fragment element access -----------------------------------------

  // One half of a packed fragment register (two halfs per .f16x2, low first).
  std::string frag_half(const std::string& word, int which) {
    const std::string z = fresh("%hw");
    if (which == 0) {
      line("  " + z + " = and i32 " + word + ", 65535");
    } else {
      const std::string t = fresh("%hw");
      line("  " + t + " = lshr i32 " + word + ", 16");
      line("  " + z + " = and i32 " + t + ", 65535");
    }
    const std::string h = fresh("%hh");
    line("  " + h + " = trunc i32 " + z + " to i16");
    return h;
  }

  // One byte of a packed .b32 fragment register (e4m3/e5m2/u8/s8: four).
  // The mask applies to every position, including byte 0: the word carries
  // three other elements alongside, and returning it whole would multiply by
  // the packed word instead of the element (caught by the ones-forms: the
  // lane tests passed by accident, their high bytes being zero).
  std::string frag_byte(const std::string& word, int which) {
    if (which == 0) return i32op("and", word, "255");
    const std::string sh = fresh("%bw");
    line("  " + sh + " = lshr i32 " + word + ", " + std::to_string(8 * which));
    return i32op("and", sh, "255");
  }

  // A/B element as f32. Half formats go through the host runtime (same code as
  // the interpreter, so the paths are bit-identical by construction); the
  // narrow floats through the shared branchless decoder.
  std::string decode_ab_f(const std::string& word, const std::string& type,
                          int within, bool perByte) {
    if (!perByte) {
      const std::string h = frag_half(word, within);
      const std::string f = fresh("%ab");
      line("  " + f + " = call float @rouge_" +
           std::string(type == "bf16" ? "bf16_to_f32" : "f16_to_f32") + "(i16 " +
           h + ")");
      return f;
    }
    const std::string b = frag_byte(word, within);
    NarrowFmt NF;
    narrow_fmt(type == "e4m3" ? "e4m3x2" : "e5m2x2", &NF);
    return emit_fp_decode(b, NF);
  }

  // A/B element as an exact i32 (integer mma: |a|,|b| <= 255).
  std::string decode_ab_i32(const std::string& word, const std::string& type,
                            int within) {
    const std::string b = frag_byte(word, within);
    if (type != "s8") return b;  // u8: the masked byte is the value
    const std::string t = fresh("%b8");
    line("  " + t + " = trunc i32 " + b + " to i8");
    const std::string s = fresh("%is");
    line("  " + s + " = sext i8 " + t + " to i32");
    return s;
  }

  // ---- mma.sync ---------------------------------------------------------

  // Where a fragment element lives, transcribed from the interpreter's gather
  // (PTX ISA 9.7.16.5.8): register index, element inside it, and the lane that
  // owns it — groupID g = lane>>2 owns rows r%8 and r%8+8, tigid = lane&3 owns
  // the column group. Everything but the row/column OFFSET relative to this
  // lane's own g/t is a compile-time constant, so no tile arrays are needed.
  struct MmaFrag {
    int K = 0, nA = 0, nB = 0, nD = 0;
    bool fp16AB = false, isInt = false, dF16 = false;
    bool perByte = false;  // A/B registers hold four bytes, not two halfs
    // A element (rOff in {0,8}, k): register, element inside it, and the
    // compile-time part of the owning lane. The full owner is g*4 + laneConst,
    // where g = lane>>2 is this lane's own row group (a runtime SSA value):
    // rows repeat every 8, so r%8 = g for both r = g and r = g+8.
    void aLoc(int rOff, int k, int* reg, int* within,
              int* laneConst) const {
      if (fp16AB && K == 16) {
        const int s = (k >= 8 ? 4 : 0) + (rOff >= 8 ? 2 : 0) + (k & 1);
        *reg = s / 2;
        *within = s % 2;
        *laneConst = (k >> 1) & 3;
      } else if (K == 16) {
        const int s = (rOff >= 8 ? 4 : 0) + (k & 3);
        *reg = s / 4;
        *within = s % 4;
        *laneConst = (k >> 2) & 3;
      } else {
        const int s = (rOff >= 8 ? 2 : 0) + (k & 1);
        *reg = s / 2;
        *within = s % 2;
        *laneConst = (k >> 1) & 3;
      }
    }
    // B element (k, col): register, element inside it, and the compile-time
    // part of the owning lane. The full owner is col*4 + laneConst, where col
    // is this lane's own column (a runtime SSA value).
    void bLoc(int k, int* reg, int* within, int* laneConst) const {
      if (fp16AB && K == 16) {
        const int s = (k >= 8 ? 2 : 0) + (k & 1);
        *reg = s / 2;
        *within = s % 2;
        *laneConst = (k >> 1) & 3;
      } else if (K == 16) {
        *reg = 0;  // a single .b32 holds all four bytes
        *within = k & 3;
        *laneConst = (k >> 2) & 3;
      } else {
        const int s = k & 1;
        *reg = 0;
        *within = s;
        *laneConst = (k >> 1) & 3;
      }
    }
    // C/D element (rOff in {0,8}, cBit in {0,1}): index inside the fragment.
    int cElem(int rOff, int cBit) const {
      return (rOff >= 8 ? 2 : 0) + cBit;
    }
  };

  bool emit_mma(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    // Split "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32".
    std::string shape, alayout, blayout, dtype, atype, btype, ctype;
    {
      std::vector<std::string> q;
      std::stringstream ss(ins.op);
      std::string t;
      while (std::getline(ss, t, '.')) q.push_back(t);
      for (const auto& s : q) {
        if (s == "mma" || s == "sync" || s == "aligned") continue;
        if (s.rfind("m", 0) == 0 && s.find("n") != std::string::npos &&
            shape.empty())
          shape = s;
        else if ((s == "row" || s == "col") && alayout.empty()) alayout = s;
        else if ((s == "row" || s == "col") && blayout.empty()) blayout = s;
        else if (dtype.empty()) dtype = s;
        else if (atype.empty()) atype = s;
        else if (btype.empty()) btype = s;
        else if (ctype.empty()) ctype = s;
      }
    }
    const bool okShape = shape == "m16n8k16" || shape == "m16n8k8";
    const bool okLayout = alayout == "row" && blayout == "col";
    const bool okAB = (atype == "f16" || atype == "bf16" || atype == "e4m3" ||
                       atype == "e5m2" || atype == "u8" || atype == "s8") &&
                      atype == btype;
    const bool okDC = (dtype == "f32" || dtype == "f16" || dtype == "s32") &&
                      dtype == ctype;
    const bool isInt = atype == "u8" || atype == "s8";
    const bool okMix = (!isInt && dtype != "s32") || (isInt && dtype == "s32");
    // m16n8k8 exists for f16/bf16 only; the byte-packed forms are k16.
    const bool okShapeType =
        (shape == "m16n8k8" && (atype == "f16" || atype == "bf16")) ||
        shape == "m16n8k16";
    if (!okShape || !okLayout || !okAB || !okDC || !okMix || !okShapeType)
      return fail("unsupported PTX op '" + ins.op + "' @ " + std::to_string(idx) +
                  " (supported: .row.col m16n8k16 with "
                  "f16/bf16/e4m3/e5m2/u8/s8 -> f32/f16/s32, and m16n8k8 with"
                  " f16/bf16 -> f32/f16)");
    MmaFrag F;
    F.K = (shape == "m16n8k16") ? 16 : 8;
    F.fp16AB = (atype == "f16" || atype == "bf16");
    F.isInt = isInt;
    F.dF16 = (dtype == "f16");
    F.perByte = !F.fp16AB;
    F.nA = (shape == "m16n8k16") ? (F.fp16AB ? 4 : 2) : 2;
    F.nB = (shape == "m16n8k16") ? (F.fp16AB ? 2 : 1) : 1;
    F.nD = F.dF16 ? 2 : 4;
    if (F.nA + F.nB > kExMaxFrag)
      return fail("mma fragments need " + std::to_string(F.nA + F.nB) +
                  " exchange words, the slot holds " +
                  std::to_string(kExMaxFrag) + " @ " + std::to_string(idx));

    const auto groups = brace_groups(A);
    if (groups.size() != 4 || static_cast<int>(groups[0].size()) != F.nD ||
        static_cast<int>(groups[1].size()) != F.nA ||
        static_cast<int>(groups[2].size()) != F.nB ||
        static_cast<int>(groups[3].size()) != F.nD)
      return fail("bad mma fragment groups in '" + ins.op + "' @ " +
                  std::to_string(idx) + " (want {" + std::to_string(F.nD) +
                  "}, {" + std::to_string(F.nA) + "}, {" + std::to_string(F.nB) +
                  "}, {" + std::to_string(F.nD) + "})");

    // On a GPU the whole tile exchange is one hardware instruction: the lane's
    // own registers go straight into it, no exchange region involved.
    if (gpu_kernel()) return emit_mma_nvptx(ins, groups);

    // Publish A and B, then read the peers. C and D stay in registers.
    // The exchange context (base + warp + own slot) is computed once here;
    // every peer read below addresses by offset from it.
    const ExchangeCtx x = exchange_ctx();
    for (int i = 0; i < F.nA; ++i) publish_slot_ctx(x, groups[1][i], i);
    for (int i = 0; i < F.nB; ++i) publish_slot_ctx(x, groups[2][i], F.nA + i);
    line("  call void @__rouge_syncthreads(ptr %launch)");

    // This lane's D geometry: g = lane>>2 (row group), t = lane&3 (column
    // group), so its four outputs are rows g/g+8 x columns 2t/2t+1.
    const std::string lane = lane_index();
    const std::string g0 = i32op("lshr", lane, "2");
    const std::string t0 = i32op("and", lane, "3");
    const std::string cPair = i32op("mul", t0, "2");

    // Owner lanes are runtime values: A element (r,k) lives in lane
    // g*4 + aConst (g = lane>>2), B element (k,c) in lane col*4 + bConst.
    const std::string g4 = i32op("shl", g0, "2");
    const std::string col4 = i32op("shl", cPair, "2");

    for (int s = 0; s < 4; ++s) {
      const int rOff = (s < 2) ? 0 : 8;
      const int cBit = s & 1;
      // This output's column: 2t + cBit. col*4 below must use the full
      // column (not col&7: columns are 0..7 here, so the mask is a no-op,
      // but the shift has to apply to the pair base plus the bit).
      const std::string colBit =
          cBit ? i32op("add", cPair, "1") : cPair;
      const std::string col4s = cBit ? i32op("shl", colBit, "2") : col4;
      // C element from this lane's own C fragment.
      std::string acc;
      if (F.isInt) {
        acc = operand(groups[3][F.cElem(rOff, cBit)]);
      } else if (F.dF16) {
        const std::string h = frag_half(operand(groups[3][s / 2]), s % 2);
        const std::string f = fresh("%cf");
        line("  " + f + " = call float @rouge_f16_to_f32(i16 " + h + ")");
        acc = f;
      } else {
        acc = bitsf32(operand(groups[3][s]));
      }
      // k-ascending accumulation, the same order the interpreter uses.
      for (int k = 0; k < F.K; ++k) {
        int aReg = 0, aWithin = 0, aConst = 0;
        F.aLoc(rOff, k, &aReg, &aWithin, &aConst);
        int bReg = 0, bWithin = 0, bConst = 0;
        F.bLoc(k, &bReg, &bWithin, &bConst);
        const std::string aLane = i32op("add", g4, std::to_string(aConst));
        const std::string bLane = i32op("add", col4s, std::to_string(bConst));
        if (F.isInt) {
          const std::string a =
              decode_ab_i32(read_frag_ctx(x, aLane, aReg), atype, aWithin);
          const std::string b = decode_ab_i32(
              read_frag_ctx(x, bLane, F.nA + bReg), btype, bWithin);
          const std::string p = i32op("mul", a, b);
          acc = i32op("add", acc, p);
          continue;
        }
        const std::string a = decode_ab_f(read_frag_ctx(x, aLane, aReg),
                                          atype, aWithin, F.perByte);
        const std::string b =
            decode_ab_f(read_frag_ctx(x, bLane, F.nA + bReg), btype, bWithin,
                        F.perByte);
        const std::string f = fresh("%mf");
        line("  " + f + " = call float @llvm.fma.f32(float " + a + ", float " + b +
             ", float " + acc + ")");
        acc = f;
      }
      // Scatter the D element.
      if (F.dF16) {
        const std::string h = fresh("%dh");
        line("  " + h + " = call i16 @rouge_f32_to_f16_rne(float " + acc + ")");
        const std::string z = fresh("%dz");
        line("  " + z + " = zext i16 " + h + " to i32");
        if (s % 2 == 0) {
          store_reg(groups[0][s / 2], z);
        } else {
          // Second half of the packed .f16x2 register: read-modify-write so
          // the half just stored by the previous iteration survives.
          const std::string prev = load_reg(groups[0][s / 2]);
          const std::string sh = i32op("shl", z, "16");
          const std::string keep = i32op("and", prev, "65535");
          store_reg(groups[0][s / 2], i32op("or", keep, sh));
        }
        continue;
      }
      store_reg(groups[0][s], F.isInt ? acc : f32bits(acc));
    }
    // The next collective in this kernel would publish into the same slots
    // while slower lanes are still reading this one's words. A second barrier
    // closes that window (the PTX has only one, because on hardware the warp
    // retires in lockstep; here lanes are OS threads).
    line("  call void @__rouge_syncthreads(ptr %launch)");
    return true;
  }

  // ---- ldmatrix / stmatrix ---------------------------------------------

  // One or more 8x8 .b16 matrices collectively moved between shared memory and
  // registers (PTX ISA 9.7.16.5.15/16). Eight threads per matrix publish the
  // row-start addresses (lanes 0-7 matrix 0, 8-15 matrix 1, ...), a group of
  // four consecutive threads moves one row, and EVERY thread receives a
  // fragment of EVERY matrix: thread t takes row t%8, elements (t%4)*2 and +1.
  // .trans swaps the receive side to columns while the addresses keep naming
  // rows, so coverage stays complete.
  bool emit_matrix_move(const rouge::PtxInstruction& ins, size_t idx) {
    const bool isLoad = ins.op[0] == 'l';
    std::string shape, num, type;
    bool trans = false, hasShared = false;
    {
      std::vector<std::string> q;
      std::stringstream ss(ins.op);
      std::string t;
      while (std::getline(ss, t, '.')) q.push_back(t);
      for (const auto& s : q) {
        if (s == "ldmatrix" || s == "stmatrix" || s == "sync" || s == "aligned")
          continue;
        else if (s == "m8n8") shape = s;
        else if (s == "x1" || s == "x2" || s == "x4") num = s;
        else if (s == "trans") trans = true;
        else if (s == "shared" || s == "shared::cta") {
          hasShared = true;
          continue;
        } else if (s == "b16") type = s;
        else
          return fail("unsupported PTX op '" + ins.op + "' @ " +
                      std::to_string(idx) + " (qualifier '" + s + "' is not"
                      " lowered)");
      }
    }
    if (shape != "m8n8" || type != "b16" ||
        (num != "x1" && num != "x2" && num != "x4"))
      return fail("unsupported PTX op '" + ins.op + "' @ " + std::to_string(idx) +
                  " (supported: .m8n8.x{1,2,4}[.trans].b16)");
    const int nMat = (num == "x4") ? 4 : (num == "x2") ? 2 : 1;
    const auto groups = brace_groups(ins.args);
    if (groups.size() != 2 ||
        static_cast<int>(groups[isLoad ? 0 : 1].size()) != nMat)
      return fail("bad " + ins.op + " operands @ " + std::to_string(idx));
    const std::vector<std::string>& fragRegs = groups[isLoad ? 0 : 1];
    // Address operand: [reg], [reg+off] or [shared symbol].
    const std::string inner = strip_brackets(groups[isLoad ? 1 : 0][0]);
    std::string addrBase;
    int64_t addrOff = 0;
    {
      std::string base;
      int64_t off = 0;
      if (split_reg_offset(inner, &base, &off)) {
        addrBase = base;
        addrOff = off;
      } else {
        addrBase = inner;
      }
    }
    std::string rowAddr;
    if (is_reg(addrBase)) {
      // A 32-bit address register is legal PTX (nvcc materialises a demoted
      // local array's address that way), and the exchange slot holds an i64:
      // widen explicitly, or the store would carry the wrong width.
      std::string v = operand(addrBase);
      if (reg_type(addrBase) != "i64") {
        const std::string t = reg_type(addrBase);
        const std::string z = fresh("%aw");
        line("  " + z + " = zext " + t + " " + v + " to i64");
        v = z;
      }
      rowAddr = v;
    } else {
      // A bare .shared/.local symbol resolves exactly as it does for ld/st.
      const std::string p = resolve_addr(groups[isLoad ? 1 : 0][0], "shared");
      const std::string v = fresh("%pi");
      line("  " + v + " = ptrtoint " + pt(p) + " " + p + " to i64");
      rowAddr = v;
    }
    if (addrOff != 0) {
      const std::string a = fresh("%ra");
      line("  " + a + " = add i64 " + rowAddr + ", " + std::to_string(addrOff));
      rowAddr = a;
    }
    // On a GPU the lane's own row address goes straight into the hardware
    // instruction: no other lane's data is needed.
    if (gpu_kernel())
      return emit_matrix_move_nvptx(ins, idx, isLoad, fragRegs, rowAddr, shape,
                                    num, trans, hasShared, type);
    const ExchangeCtx x = exchange_ctx();
    publish_row_addr_ctx(x, rowAddr);
    line("  call void @__rouge_syncthreads(ptr %launch)");

    const std::string lane = lane_index();
    // Thread t takes row t>>2, elements (t%4)*2 and +1 (.trans: column t>>2,
    // rows (t%4)*2) — a group of four consecutive threads moves one whole row.
    const std::string fixed = i32op("lshr", lane, "2");
    const std::string e0 = i32op("mul", i32op("and", lane, "3"), "2");
    for (int m = 0; m < nMat; ++m) {
      std::string packed;
      for (int e = 0; e < 2; ++e) {
        // .trans: the thread takes column `fixed', rows e0+e. Plain: row
        // `fixed', columns e0+e. Both read the address the row's owner
        // published, so coverage is complete either way.
        const std::string r =
            trans ? i32op("add", e0, std::to_string(e)) : fixed;
        const std::string c =
            trans ? fixed : i32op("add", e0, std::to_string(e));
        const std::string provider = i32op("add", std::to_string(m * 8), r);
        const std::string base = read_peer_row_ctx(x, provider);
        const std::string byteOff =
            i32op("mul", c, "2");  // i32 column index -> i32 byte count
        const std::string bo64 = fresh("%bo");
        line("  " + bo64 + " = zext i32 " + byteOff + " to i64");
        const std::string at = fresh("%at");
        line("  " + at + " = add i64 " + base + ", " + bo64);
        const std::string shp = inttoptr_as(at, shared_ptr_ty());
        if (isLoad) {
          const std::string v = fresh("%mv");
          line("  " + v + " = load i16, " + pt(shp) + " " + shp + ", align 2");
          const std::string z = fresh("%mz");
          line("  " + z + " = zext i16 " + v + " to i32");
          packed = (e == 0) ? z : i32op("or", packed, i32op("shl", z, "16"));
        } else {
          const std::string h = frag_half(operand(fragRegs[m]), e);
          line("  store i16 " + h + ", " + pt(shp) + " " + shp + ", align 2");
        }
      }
      if (isLoad) store_reg(fragRegs[m], packed);
    }
    // See emit_mma: the next collective must not overwrite the published
    // addresses before every lane has read them.
    line("  call void @__rouge_syncthreads(ptr %launch)");
    return true;
  }

  // ---- mma.sync on NVPTX --------------------------------------------------
  //
  // One hardware instruction reached through inline assembly (NVVM exposes no
  // ldmatrix/mma.sync intrinsics in this form: probed against clang, only
  // inline asm lowers). The template reuses the validated PTX mnemonic
  // verbatim; every fragment word is an i32 SSA (f16x2/f32/s32 payloads are
  // all 32-bit patterns here). sideeffect keeps the warp-scoped ordering
  // against surrounding memory operations.
  bool emit_mma_nvptx(
      const rouge::PtxInstruction& ins,
      const std::vector<std::vector<std::string>>& groups) {
    // Input words first: operand() emits its loads before the call.
    std::vector<std::string> words;
    for (size_t g = 1; g < groups.size(); ++g)
      for (const auto& r : groups[g]) words.push_back(operand(r));
    // "{d0,d1}, {a0,..}, {b0,..}, {c0,..}" with one $i per operand in group
    // order. PTX requires the four groups spelled separately (a single
    // 10-wide second group is rejected: "Illegal vector size").
    std::string frags, cons;
    size_t k = 0;
    for (size_t g = 0; g < groups.size(); ++g) {
      if (g) frags += ", ";
      frags += "{";
      for (size_t i = 0; i < groups[g].size(); ++i) {
        if (i) {
          frags += ",";
          cons += ",";
        } else if (g) {
          cons += ",";
        }
        frags += "$" + std::to_string(k++);
        cons += (g == 0 ? "=r" : "r");
      }
      frags += "}";
    }
    frags += ";";
    // The call returns one i32 per D word as an aggregate.
    std::string retty = "{ ";
    for (size_t i = 0; i < groups[0].size(); ++i) {
      if (i) retty += ", ";
      retty += "i32";
    }
    retty += " }";
    std::string call = "call " + retty + " asm sideeffect \"" + ins.op + " " +
                       frags + "\", \"" + cons + "\"(";
    for (size_t i = 0; i < words.size(); ++i) {
      if (i) call += ", ";
      call += "i32 " + words[i];
    }
    call += ")";
    const std::string agg = fresh("%mm");
    line("  " + agg + " = " + call);
    for (size_t i = 0; i < groups[0].size(); ++i) {
      const std::string e = fresh("%md");
      line("  " + e + " = extractvalue " + retty + " " + agg + ", " +
           std::to_string(i));
      store_reg(groups[0][i], e);
    }
    return true;
  }

  // ---- ldmatrix / stmatrix on NVPTX --------------------------------------
  //
  // The lane's own row-start address goes straight into the hardware
  // instruction (other lanes' registers are the hardware's business, not
  // ours). Template is the canonical mnemonic rebuilt from the validated
  // qualifiers; the address is the same absolute shared-window i64 the host
  // path publishes. sideeffect keeps ordering against surrounding accesses.
  bool emit_matrix_move_nvptx(const rouge::PtxInstruction& ins, size_t idx,
                              bool isLoad,
                              const std::vector<std::string>& fragRegs,
                              const std::string& rowAddr,
                              const std::string& shape, const std::string& num,
                              bool trans, bool hasShared,
                              const std::string& type) {
    (void)idx;
    const int nMat = (int)fragRegs.size();
    std::string tmpl = (isLoad ? std::string("ldmatrix") : std::string("stmatrix")) +
                       ".sync.aligned." + shape + "." + num +
                       (trans ? ".trans" : "") + (hasShared ? ".shared" : "") +
                       "." + type + " ";
    std::string cons;
    if (isLoad) {
      tmpl += "{";
      for (int i = 0; i < nMat; ++i) {
        if (i) {
          tmpl += ",";
          cons += ",";
        }
        tmpl += "$" + std::to_string(i);
        cons += "=r";
      }
      tmpl += "}, [$" + std::to_string(nMat) + "];";
      cons += ",l";
      std::string retty = nMat == 1 ? "i32" : "{ ";
      if (nMat > 1) {
        for (int i = 0; i < nMat; ++i) {
          if (i) retty += ", ";
          retty += "i32";
        }
        retty += " }";
      }
      const std::string v = fresh("%mx");
      line("  " + v + " = call " + retty + " asm sideeffect \"" + tmpl +
           "\", \"" + cons + "\"(i64 " + rowAddr + ")");
      if (nMat == 1) {
        store_reg(fragRegs[0], v);
      } else {
        for (int i = 0; i < nMat; ++i) {
          const std::string e = fresh("%me");
          line("  " + e + " = extractvalue " + retty + " " + v + ", " +
               std::to_string(i));
          store_reg(fragRegs[i], e);
        }
      }
      return true;
    }
    tmpl += "[$0], {";
    cons = "l";
    std::string args = "(i64 " + rowAddr;
    for (int i = 0; i < nMat; ++i) {
      tmpl += (i ? "," : "") + std::string("$") + std::to_string(i + 1);
      cons += ",r";
      args += ", i32 " + operand(fragRegs[i]);
    }
    tmpl += "};";
    args += ")";
    line("  call void asm sideeffect \"" + tmpl + "\", \"" + cons + "\"" +
         args);
    return true;
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
    if (!is_reg(tok) || reg_type(tok) != ty) return val;
    const std::string m = fresh("%sh");
    line("  " + m + " = and " + ty + " " + val + ", " + std::to_string(mask));
    return m;
  }

  // and.* / or.* / xor.* / not.* / shl.* / shr.* on 16-, 32- and 64-bit types.
  bool emit_alu_logic(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    if (!check_reg_width(ins, idx, op)) return false;
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

  // A typed ALU instruction (add.s32, mul.lo.s32, ...) must write a register
  // of exactly that width. A mismatch in the DESTINATION (e.g.
  // "add.u32 %rd6, %rd6, 0" on a 64-bit register) is invalid PTX and produced
  // IR that computed something plausible on the host and something else on
  // the card; refuse it loudly.
  //
  // A narrower SOURCE is accepted: our hand-written canonical kernels use
  // "add.s64 %rd4, %rd2, %r7" (a byte offset widened on the fly), and both
  // the interpreter and this lowering have always read such a source as
  // zero-extended. The two paths agree, so the extension is consistent;
  // it is a deviation from the ISA, and it is documented as one.
  bool check_reg_width(const rouge::PtxInstruction& ins, size_t idx,
                       const std::string& op) {
    const int want = int_type_bits(op);
    if (want == 0) return true;
    if (ins.args.empty()) return true;
    const std::string dst = strip_brackets(clean_operand(ins.args[0]));
    if (!is_reg(dst)) return true;
    const int have = reg_bits(dst);
    if (have != 0 && have != want) {
      fail("'" + op + "' expects a " + std::to_string(want) +
           "-bit destination, but '" + dst + "' is " + std::to_string(have) +
           "-bit @ " + std::to_string(idx));
      return false;
    }
    return true;
  }

  // mul.lo / mul.hi / mad.lo / mad.hi / div / rem on 32- and 64-bit types.
  bool emit_alu_muldiv(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    if (!check_reg_width(ins, idx, op)) return false;
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
    // The symbol's own address, not a load through it: "@g = external global
    // [N x i8]" names the storage itself, so "load ptr, ptr @g" would read the
    // first eight bytes OF the array as if they were a pointer. On the host
    // that silently produced garbage; on a GPU it aimed an atomic at address
    // zero and faulted (found by executing real nvcc output on hardware).
    const std::string i = fresh("%gi");
    line("  " + i + " = ptrtoint ptr @" + sym + " to i64");
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

  // prmt.b32 — byte permute. nvcc uses it to pack bytes (four byte loads
  // into one 32-bit store is the typical case), so real PTX needs it.
  // Semantics (PTX ISA 9.7.10.7): the eight source bytes are the 64-bit
  // value {b, a} counted from the low end (index 0..3 are a's bytes, 4..7
  // are b's). Each 4-bit field of c selects the source byte for d.b0..d.b3
  // (field 0 -> d.b0); the field's msb asks for the byte's SIGN to be
  // replicated over the whole 8-bit target position.
  //
  // Lowered without the llvm.ptx.prmt intrinsic on purpose: that intrinsic
  // exists only for the NVPTX target, while the same .ll must also build and
  // run on the host. The explicit form is portable and matches the
  // interpreter bit for bit.
  bool emit_prmt(const rouge::PtxInstruction& ins, size_t idx) {
    (void)idx;
    const auto& A = ins.args;
    // src64 = a | (b << 32): byte 0..3 are a's bytes, 4..7 are b's.
    const std::string a32 = zext_to_i64(operand(A[1]), reg_type(A[1]));
    const std::string b32 = zext_to_i64(operand(A[2]), reg_type(A[2]));
    const std::string bsh = fresh("%pb");
    line("  " + bsh + " = shl i64 " + b32 + ", 32");
    const std::string src = fresh("%ps");
    line("  " + src + " = or i64 " + a32 + ", " + bsh);
    const std::string sel = zext_to_i64(operand(A[3]), reg_type(A[3]));
    std::string acc;
    for (int i = 0; i < 4; ++i) {
      // field i of c: (c >> 4i) & 0xF
      const std::string sh = fresh("%ph");
      line("  " + sh + " = lshr i64 " + sel + ", " + std::to_string(4 * i));
      const std::string m = fresh("%pm");
      line("  " + m + " = and i64 " + sh + ", 15");
      // index = field & 7; sign bit = field & 8.
      const std::string idx = fresh("%pi");
      line("  " + idx + " = and i64 " + m + ", 7");
      const std::string bs = fresh("%pb2");
      line("  " + bs + " = shl i64 " + idx + ", 3");
      const std::string v = fresh("%pv");
      line("  " + v + " = lshr i64 " + src + ", " + bs);
      const std::string v8 = fresh("%pv8");
      line("  " + v8 + " = and i64 " + v + ", 255");
      // msb of the field: replicate the sign of the selected byte.
      const std::string sbit = fresh("%psb");
      line("  " + sbit + " = and i64 " + m + ", 8");
      const std::string isneg = fresh("%pn");
      line("  " + isneg + " = icmp ne i64 " + sbit + ", 0");
      const std::string top = fresh("%pt");
      line("  " + top + " = and i64 " + v8 + ", 128");
      const std::string topne = fresh("%ptn");
      line("  " + topne + " = icmp ne i64 " + top + ", 0");
      const std::string rep = fresh("%pr");
      line("  " + rep + " = select i1 " + topne + ", i64 255, i64 0");
      const std::string lit = fresh("%pl");
      line("  " + lit + " = select i1 " + isneg + ", i64 " + rep + ", i64 " + v8);
      if (i == 0) {
        acc = lit;
      } else {
        const std::string sh2 = fresh("%pq");
        line("  " + sh2 + " = shl i64 " + lit + ", " + std::to_string(8 * i));
        const std::string o = fresh("%po");
        line("  " + o + " = or i64 " + acc + ", " + sh2);
        acc = o;
      }
    }
    const std::string res = fresh("%r");
    line("  " + res + " = trunc i64 " + acc + " to i32");
    store_reg(A[0], res);
    return true;
  }

  // dp4a / dp2a — dot product of byte pairs. nvcc emits dp4a for int8 dot
  // products, i.e. exactly the inner loop of an int8 GEMM.
  //   dp4a.atype.btype d, a, b, c : Va = the four bytes of a (per .atype),
  //     Vb = the four bytes of b (per .btype), d = c + sum Va[i]*Vb[i].
  //   dp2a.mode.atype.btype d, a, b, c : Va = the two 16-bit halves of a,
  //     Vb = the four bytes of b, b_select = (.lo ? 0 : 2),
  //     d = c + Va[0]*Vb[b_select] + Va[1]*Vb[b_select+1].
  // Lowered with explicit byte extraction so the same .ll builds for the
  // host as well (llvm.ptx.dp4a exists only for NVPTX).
  bool emit_dot4a(const rouge::PtxInstruction& ins, size_t idx) {
    (void)idx;
    const auto& A = ins.args;
    const std::string op = ins.op;
    const bool wide = op.rfind("dp4a.", 0) == 0;
    const bool use_low = wide || (op.find(".lo.") != std::string::npos);
    // Tokens: dp4a.atype.btype | dp2a.mode.atype.btype
    std::vector<std::string> tok;
    {
      std::string cur;
      for (char ch : op) {
        if (ch == '.') { tok.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
      }
      tok.push_back(cur);
    }
    const size_t at_idx = wide ? 1u : 2u;
    const std::string atype = tok.size() > at_idx ? tok[at_idx] : std::string("u32");
    const std::string btype = tok.size() > at_idx + 1u ? tok[at_idx + 1u] : std::string("u32");
    const bool a_signed = atype == "s32";
    const bool b_signed = btype == "s32";
    const std::string a = zext_to_i64(operand(A[1]), reg_type(A[1]));
    const std::string b = zext_to_i64(operand(A[2]), reg_type(A[2]));
    const std::string c = zext_to_i64(operand(A[3]), reg_type(A[3]));
    std::string sum;
    if (wide) {
      for (int i = 0; i < 4; ++i) {
        const std::string av = extract_lane(a, 8 * i, 8, a_signed);
        const std::string bv = extract_lane(b, 8 * i, 8, b_signed);
        sum = accumulate_product(sum, av, bv);
      }
    } else {
      for (int i = 0; i < 2; ++i) {
        const std::string av = extract_lane(a, 16 * i, 16, a_signed);
        const std::string bv = extract_lane(b, 8 * (i + (use_low ? 0 : 2)), 8, b_signed);
        sum = accumulate_product(sum, av, bv);
      }
    }
    const std::string tot = fresh("%dt");
    line("  " + tot + " = add i64 " + c + ", " + sum);
    const std::string res = fresh("%r");
    line("  " + res + " = trunc i64 " + tot + " to i32");
    store_reg(A[0], res);
    return true;
  }

  // Extract a `bits`-wide lane from a 64-bit value at the given bit offset and
  // sign/zero extend it to 64 bits.
  std::string extract_lane(const std::string& v, int off, int bits, bool is_signed) {
    const std::string sh = fresh("%dl");
    line("  " + sh + " = lshr i64 " + v + ", " + std::to_string(off));
    const std::string m = fresh("%dm");
    line("  " + m + " = and i64 " + sh + ", " + std::to_string((1LL << bits) - 1));
    const std::string out = fresh("%dv");
    if (is_signed) {
      const std::string t = fresh("%dt2");
      line("  " + t + " = shl i64 " + m + ", " + std::to_string(64 - bits));
      line("  " + out + " = ashr i64 " + t + ", " + std::to_string(64 - bits));
    } else {
      line("  " + out + " = or i64 " + m + ", 0");
    }
    return out;
  }

  // sum = sum + x*y (sum empty -> the first product).
  std::string accumulate_product(const std::string& sum, const std::string& x,
                                 const std::string& y) {
    const std::string p = fresh("%dp");
    line("  " + p + " = mul i64 " + x + ", " + y);
    if (sum.empty()) return p;
    const std::string o = fresh("%do");
    line("  " + o + " = add i64 " + sum + ", " + p);
    return o;
  }

  bool emit_alu_unary(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    const std::string op = ins.op;
    if (!check_reg_width(ins, idx, op)) return false;
    if ((A.size() == 4 || A.size() == 5) &&
        (op.rfind("bfi.", 0) == 0 || op.rfind("bfe.", 0) == 0))
      return emit_bitfield(ins, idx);
    if (A.size() == 4 && op.rfind("prmt.", 0) == 0)
      return emit_prmt(ins, idx);
    if (A.size() == 4 && (op.rfind("dp4a.", 0) == 0 || op.rfind("dp2a.", 0) == 0))
      return emit_dot4a(ins, idx);
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
    if (!check_reg_width(ins, idx, op)) return false;

    // f16x2/bf16x2 fallback: keep payload as i32 (vector of 2×half).
    // Real packed arithmetic will be vectorized in MLIR stage 2; for AOT
    // correctness just preserve bit pattern (or do i32 add) so codegen does not fail.
    // Matched on the full type names: a bare "x2" also occurs in matrix
    // multiplicities (ldmatrix.x2), which must never take this fallback.
    if (op.find(".f16x2") != std::string::npos ||
        op.find(".bf16x2") != std::string::npos) {
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
        if (is_reg(a) && reg_bits(a) != 16)
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
      // fma.rn.f32 is a 4-operand single-rounding op: routing it into the
      // fdiv below (which reads only two operands) silently miscompiled
      // every fma into a division (matrixMul caught this on hardware).
      if (op.rfind("fma.", 0) == 0) {
        if (A.size() != 4)
          return fail("bad " + op + " operands @ " + std::to_string(idx));
        const std::string x = load_f32(A[1]);
        const std::string y = load_f32(A[2]);
        const std::string z = load_f32(A[3]);
        const std::string r = fresh("%r");
        line("  " + r + " = call float @llvm.fma.f32(float " + x + ", float " +
             y + ", float " + z + ")");
        store_f32(A[0], r);
        return true;
      }
      if (A.size() != 3)
        return fail("bad " + op + " operands @ " + std::to_string(idx));
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
    // Integer: u32/s32/u64/s64. Both operands are fitted to the destination
    // width: PTX allows a narrower register as a source of a wider arithmetic
    // op ("add.s64 %rd2, %rd1, %r9" is what nvcc emits for pointer maths), and
    // the interpreter reads such a register zero-extended. Without the fit the
    // emitted IR mixed i32 into an i64 expression and clang rejected it.
    const std::string ty = reg_type(A[0]);
    const char* ll = op[0] == 'a' ? "add" : op[0] == 's' ? "sub" : "mul";
    const auto fit_src = [&](const std::string& tok) {
      std::string v = operand(tok);
      const std::string srcTy =
          is_reg(tok) ? reg_type(tok)
                      : (int_type_bits(op) == 64 ? "i64" : "i32");
      if (srcTy == ty) return v;
      const std::string t = fresh("%t");
      line("  " + t + " = " + (type_bits(ty) > type_bits(srcTy) ? "zext" : "trunc") +
           " " + srcTy + " " + v + " to " + ty);
      return t;
    };
    const std::string x = fit_src(A[1]);
    const std::string y = fit_src(A[2]);
    const std::string r = fresh("%r");
    line("  " + r + " = " + ll + " " + ty + " " + x + ", " + y);
    store_reg(A[0], r);
    // Address provenance flows through add/sub (never mul: a product of an
    // address is not an address): "add.s64 %rd22, %rd21, %rd19" keeps the
    // shared/local window of its sources, so generic accesses through the
    // derived register resolve to the right address space (clang-emitted
    // generic ld.b32 [%rd] over shared memory caught this on NVPTX).
    if (op[0] == 'a' || op[0] == 's') {
      const auto marked = [&](const std::string& t) {
        return is_reg(t) &&
               (is_shared_addr_reg(t) || is_local_addr_reg(t));
      };
      const std::string* src =
          (A.size() > 1 && marked(A[1]))
              ? &A[1]
              : (A.size() > 2 && marked(A[2]) ? &A[2] : nullptr);
      if (src != nullptr) {
        if (is_shared_addr_reg(*src)) shared_addr_regs_.insert(A[0]);
        if (is_local_addr_reg(*src)) local_addr_regs_.insert(A[0]);
      }
    }
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
      const std::string addr = resolve_addr(A[1], op_space(op));
      const std::string cmp = operand(A[2]);
      const std::string nw = operand(A[3]);
      if (wf32) {
        // f32 cas via b32 bitcast: compare as i32
        const std::string pair = fresh("%cas");
        const std::string old = fresh("%old");
        line("  " + pair + " = cmpxchg " + pt(addr) + " " + addr + ", i32 " + cmp + ", i32 " + nw +
             " monotonic monotonic, align 4");
        line("  " + old + " = extractvalue { i32, i1 } " + pair + ", 0");
        store_reg(A[0], old);
        return true;
      }
      if (w64) {
        const std::string pair = fresh("%cas");
        const std::string old = fresh("%old");
        line("  " + pair + " = cmpxchg " + pt(addr) + " " + addr + ", i64 " + cmp + ", i64 " + nw +
             " monotonic monotonic, align 8");
        line("  " + old + " = extractvalue { i64, i1 } " + pair + ", 0");
        store_reg(A[0], old);
        return true;
      }
      // b32/u32/s32
      const std::string pair = fresh("%cas");
      const std::string old = fresh("%old");
      line("  " + pair + " = cmpxchg " + pt(addr) + " " + addr + ", i32 " + cmp + ", i32 " + nw +
           " monotonic monotonic, align 4");
      line("  " + old + " = extractvalue { i32, i1 } " + pair + ", 0");
      store_reg(A[0], old);
      return true;
    }
    if (is_exch) {
      if (!has_dst) return fail("red.exch not supported '" + op + "' @ " + std::to_string(idx));
      if (A.size() < 3) return fail("bad atomic operands @ " + std::to_string(idx));
      const std::string addr = resolve_addr(A[1], op_space(op));
      const std::string val = operand(A[2]);
      const std::string old = fresh("%old");
      if (wf32) {
        // exch f32 via i32 bitcast
        line("  " + old + " = atomicrmw xchg " + pt(addr) + " " + addr + ", i32 " + val + " monotonic, align 4");
        store_reg(A[0], old);
        return true;
      }
      if (w64) {
        line("  " + old + " = atomicrmw xchg " + pt(addr) + " " + addr + ", i64 " + val + " monotonic, align 8");
      } else {
        line("  " + old + " = atomicrmw xchg " + pt(addr) + " " + addr + ", i32 " + val + " monotonic, align 4");
      }
      store_reg(A[0], old);
      return true;
    }
    if (is_min || is_max) {
      const size_t ai = has_dst ? 1 : 0;
      if (A.size() < ai + 2) return fail("bad atomic operands @ " + std::to_string(idx));
      const std::string addr = resolve_addr(A[ai], op_space(op));
      const std::string val = operand(A[ai + 1]);
      const std::string old = fresh("%old");
      if (wf32) {
        const std::string f = fresh("%f");
        line("  " + f + " = bitcast i32 " + val + " to float");
        const char* llOp = is_min ? "fmin" : "fmax";
        line("  " + old + " = atomicrmw " + llOp + " " + pt(addr) + " " + addr + ", float " + f + " monotonic, align 4");
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
        line("  " + old + " = atomicrmw " + llOp + " " + pt(addr) + " " + addr + ", double " + f + " monotonic, align 8");
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
      line("  " + old + " = atomicrmw " + llOp + " " + pt(addr) + " " + addr + ", " + llTy + " " + val +
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
    const std::string addr = resolve_addr(A[ai], op_space(op));
    const int align = w64 ? 8 : 4;
    const char* llTy = w64 ? "i64" : "i32";
    const std::string old = fresh("%old");
    if (ty == "f32") {
      const std::string f = fresh("%f");
      line("  " + f + " = bitcast i32 " + operand(A[ai + 1]) + " to float");
      line("  " + old + " = atomicrmw fadd " + pt(addr) + " " + addr + ", float " + f +
           " monotonic, align 4");
    } else if (ty == "u32" || ty == "s32" || ty == "b32" || w64) {
      line("  " + old + " = atomicrmw add " + pt(addr) + " " + addr + ", " + llTy + " " +
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

  // ---- narrow float formats (e4m3 / e5m2 / e3m2 / e2m3 / e2m1) --------------
  // Bit layouts and special values are the OCP specs (OFP8 Table 1-2, OCP MX
  // v1.0 Table 7), mirrored from the interpreter (rouge-ptx/ptx.cpp), which
  // documents the exact spec references. The IR below is branchless integer
  // arithmetic + selects, op-for-op with the C++ scalar cores, so AOT and
  // interpreter agree bit-for-bit on every input.
  //
  // LLVM IR has no nested instructions: every operation's result is bound to
  // a fresh SSA name by the helpers below (i32 arithmetic, i1 logic,
  // selects, float bitcasts).
  std::string i32op(const std::string& op, const std::string& a,
                    const std::string& b) {
    const std::string r = fresh("%b");
    line("  " + r + " = " + op + " i32 " + a + ", " + b);
    return r;
  }
  std::string i1op(const std::string& op, const std::string& a,
                   const std::string& b) {
    const std::string r = fresh("%c");
    line("  " + r + " = " + op + " i1 " + a + ", " + b);
    return r;
  }
  std::string i32cmp(const std::string& pred, const std::string& a,
                     const std::string& b) {
    const std::string r = fresh("%cc");
    line("  " + r + " = icmp " + pred + " i32 " + a + ", " + b);
    return r;
  }
  std::string i32sel(const std::string& c, const std::string& a,
                     const std::string& b) {
    const std::string r = fresh("%sl");
    line("  " + r + " = select i1 " + c + ", i32 " + a + ", i32 " + b);
    return r;
  }
  std::string f32sel(const std::string& c, const std::string& a,
                     const std::string& b) {
    const std::string r = fresh("%sl");
    line("  " + r + " = select i1 " + c + ", float " + a + ", float " + b);
    return r;
  }
  std::string f32cmp(const std::string& pred, const std::string& a,
                     const std::string& b) {
    const std::string r = fresh("%fc");
    line("  " + r + " = fcmp " + pred + " float " + a + ", " + b);
    return r;
  }
  std::string f32bits(const std::string& f) {
    const std::string r = fresh("%fb");
    line("  " + r + " = bitcast float " + f + " to i32");
    return r;
  }
  std::string bitsf32(const std::string& b) {
    const std::string r = fresh("%bf");
    line("  " + r + " = bitcast i32 " + b + " to float");
    return r;
  }
  std::string f32const(uint32_t bits) {
    const std::string r = fresh("%fc");
    line("  " + r + " = bitcast i32 " + std::to_string(bits) + " to float");
    return r;
  }
  struct NarrowFmt {
    int E, M, bias;
    uint32_t maxBits, nanBits;  // field patterns (no sign); nanBits 0 = none
    int shift, width;           // container placement (.b8; e2m1 central [5:2])
    bool nanToMax;              // NaN -> positive MAX_NORM (FP6/FP4 rule)
  };
  static bool narrow_fmt(const std::string& name, NarrowFmt* out) {
    // maxBits — MAX_NORM БЕЗ знакового бита (в том же смысле, что в
    // интерпретаторе): со знаком насыщение при положительном переполнении
    // давало -MAX вместо +MAX. Константы синхронны с интерпретатором.
    if (name == "e4m3x2") *out = {4, 3, 7, 0x7E, 0x7F, 0, 8, false};
    else if (name == "e5m2x2") *out = {5, 2, 15, 0x7B, 0x7F, 0, 8, false};
    else if (name == "e3m2x2") *out = {3, 2, 3, 0x1F, 0, 0, 6, true};
    else if (name == "e2m3x2") *out = {2, 3, 1, 0x1F, 0, 0, 6, true};
    else if (name == "e2m1x2") *out = {2, 1, 1, 0x07, 0, 2, 4, true};
    else return false;
    return true;
  }

  // f32 SSA -> narrow field bits (i32, low fieldWidth bits). Mirrors
  // f32_to_narrow() in the interpreter exactly (same order of checks).
  // f32 SSA -> narrow field bits (i32, low fieldWidth bits). Mirrors
  // f32_to_narrow() in the interpreter exactly (same order of checks).
  std::string emit_fp_encode(const std::string& floatSSA, const NarrowFmt& F,
                             bool rz, bool relu) {
    const int W = F.width;
    const uint32_t maxExp = (1u << F.E) - 1u;
    const uint32_t allMant = (1u << F.M) - 1u;
    const uint32_t capNum = (1u << (F.E + F.M)) - 1u;
    const bool topNormal =
        ((F.maxBits >> F.M) & maxExp) == maxExp;
    const std::string Sw = std::to_string(W - 1);
    const std::string SmaxExp = std::to_string(maxExp);
    // relu first (float domain, same condition as the interpreter).
    std::string v = floatSSA;
    if (relu) {
      const std::string isNan = f32cmp("uno", v, "0.0");
      const std::string isNeg = f32cmp("olt", v, "0.0");
      const std::string qnan = f32const(0x7FC00000u);
      const std::string pos0 = f32const(0u);
      v = f32sel(isNan, qnan, v);
      v = f32sel(isNeg, pos0, v);
    }
    const std::string x = f32bits(v);
    const std::string sign = i32op("lshr", x, "31");
    const std::string e = i32op("and", i32op("lshr", x, "23"), "255");
    const std::string man = i32op("and", x, "8388607");
    const std::string eMax = i32cmp("eq", e, "255");
    const std::string manNZ = i32cmp("ne", man, "0");
    const std::string isnan = i1op("and", eMax, manNZ);
    const std::string manZ = i32cmp("eq", man, "0");
    const std::string isinf = i1op("and", eMax, manZ);
    const std::string eZero = i32cmp("eq", e, "0");
    const std::string iszero = i1op("and", eZero, manZ);
    const std::string shSign = i32op("shl", sign, Sw);
    const std::string signedMax = i32op("or", shSign, std::to_string(F.maxBits));
    const uint32_t nanPat = F.nanToMax ? F.maxBits : F.nanBits;
    const std::string signedNan = i32op("or", shSign, std::to_string(nanPat));
    const std::string unbiased = i32op("sub", e, "127");
    const std::string tExp = i32op("add", unbiased, std::to_string(F.bias));
    const std::string ovf = i32cmp("sgt", tExp, SmaxExp);
    const std::string isSub = i32cmp("sle", tExp, "0");
    const std::string tiny = i32cmp("slt", tExp, std::to_string(-F.M));
    // Subnormal path.
    const int mantShift = 23 - F.M;
    const std::string full = i32op("or", i32op("shl", "1", "23"), man);
    const std::string shift = i32op("add", i32op("sub", "1", tExp),
                                    std::to_string(mantShift));
    const std::string valSub = i32op("lshr", full, shift);
    const std::string remMask = i32op("sub", i32op("shl", "1", shift), "1");
    const std::string rem = i32op("and", full, remMask);
    const std::string half = i32op("shl", "1", i32op("sub", shift, "1"));
    std::string doInc;
    if (rz) {
      doInc = i32cmp("eq", "0", "1");  // false: truncate
    } else {
      const std::string gtHalf = i32cmp("ugt", rem, half);
      const std::string eqHalf = i32cmp("eq", rem, half);
      const std::string oddLow = i32cmp("eq", i32op("and", valSub, "1"), "1");
      doInc = i1op("or", gtHalf, i1op("and", eqHalf, oddLow));
    }
    const std::string valSubR = i32op("add", valSub, i32sel(doInc, "1", "0"));
    const std::string valSubC = i32sel(i32cmp("ugt", valSubR, std::to_string(capNum)),
                                       std::to_string(capNum), valSubR);
    std::string subCarry;
    if (F.nanBits != 0) {
      subCarry = i32cmp("eq", i32op("lshr", valSubC, std::to_string(F.M)), SmaxExp);
    } else {
      subCarry = i32cmp("eq", "0", "1");  // false
    }
    const std::string subRes =
        i32sel(subCarry, signedMax,
               i32op("or", shSign, i32op("and", valSubC, std::to_string(capNum))));
    // Normal path.
    const int dropBits = 23 - F.M;
    const std::string val =
        i32op("or", i32op("shl", tExp, std::to_string(F.M)),
              i32op("lshr", man, std::to_string(dropBits)));
    const std::string remN = i32op("and", man, std::to_string((1u << dropBits) - 1u));
    const std::string halfN = i32op("shl", "1", std::to_string(dropBits - 1));
    std::string doIncN;
    if (rz) {
      doIncN = i32cmp("eq", "0", "1");
    } else {
      const std::string gtH = i32cmp("ugt", remN, halfN);
      const std::string eqH = i32cmp("eq", remN, halfN);
      const std::string oddL = i32cmp("eq", i32op("and", val, "1"), "1");
      doIncN = i1op("or", gtH, i1op("and", eqH, oddL));
    }
    const std::string valR = i32op("add", val, i32sel(doIncN, "1", "0"));
    const std::string gotExp = i32op("lshr", valR, std::to_string(F.M));
    const std::string pastMax = i32cmp("ugt", gotExp, SmaxExp);
    const std::string atMax = i32cmp("eq", gotExp, SmaxExp);
    std::string needMax;
    if (F.nanBits != 0) {
      if (!topNormal) {
        needMax = atMax;  // e5m2: anything in exp==31 saturates
      } else {
        // e4m3: only the NaN pattern (mantissa all ones) saturates.
        needMax = i1op("and", atMax,
                       i32cmp("eq", i32op("and", valR, std::to_string(allMant)),
                              std::to_string(allMant)));
      }
    } else {
      needMax = pastMax;  // no-NaN formats saturate only past the top binade
    }
    const uint32_t fMask = (F.width == 32) ? 0xffffffffu : ((1u << W) - 1u);
    const std::string normRes = i32sel(
        needMax, signedMax,
        i32op("or", shSign, i32op("and", valR, std::to_string(fMask))));
    // Priority exactly as in the interpreter.
    std::string r = i32sel(isSub, subRes, normRes);
    r = i32sel(tiny, "0", r);
    r = i32sel(ovf, signedMax, r);
    r = i32sel(iszero, shSign, r);
    r = i32sel(isinf, signedMax, r);
    r = i32sel(isnan, signedNan, r);
    return r;
  }

  // Narrow container byte (i32 SSA, low 8 bits) -> f32 SSA. Mirrors
  // narrow_to_f32() in the interpreter exactly.
  std::string emit_fp_decode(const std::string& byteSSA, const NarrowFmt& F) {
    const int W = F.width;
    const uint32_t maxExp = (1u << F.E) - 1u;
    const uint32_t allMant = (1u << F.M) - 1u;
    const bool topNormal =
        ((F.maxBits >> F.M) & maxExp) == maxExp;
    const std::string Sw = std::to_string(W - 1);
    const std::string SmaxExp = std::to_string(maxExp);
    const std::string SallMant = std::to_string(allMant);
    std::string field = byteSSA;
    if (F.shift != 0 || W != 8) {
      field = i32op("and", i32op("lshr", byteSSA, std::to_string(F.shift)),
                    std::to_string((1u << W) - 1u));
    }
    const std::string sign = i32op("lshr", field, Sw);
    const std::string exp =
        i32op("and", i32op("lshr", field, std::to_string(F.M)), SmaxExp);
    const std::string man = i32op("and", field, SallMant);
    const std::string fsign = i32op("shl", sign, "31");
    const std::string isZeroE = i32cmp("eq", exp, "0");
    const std::string isZeroM = i32cmp("eq", man, "0");
    const std::string isZero = i1op("and", isZeroE, isZeroM);
    const std::string zeroRes = bitsf32(fsign);
    // Subnormal: unrolled normalize (M steps suffice: man < 2^M).
    std::string curM = man, curS = i32op("sub", "0", "0");
    const std::string bitM = std::to_string(1u << F.M);
    for (int i = 0; i < F.M; ++i) {
      const std::string need = i32cmp("eq", i32op("and", curM, bitM), "0");
      curM = i32sel(need, i32op("shl", curM, "1"), curM);
      curS = i32sel(need, i32op("add", curS, "1"), curS);
    }
    const std::string normM = i32op("and", curM, SallMant);
    const std::string subExp =
        i32op("sub", std::to_string(127 - F.bias + 1), curS);
    const std::string subBits = i32op(
        "or", i32op("or", fsign, i32op("shl", subExp, "23")),
        i32op("shl", i32op("and", normM, SallMant), std::to_string(23 - F.M)));
    const std::string subRes = bitsf32(subBits);
    // Top exponent: per-format rule.
    const std::string isTop = i32cmp("eq", exp, SmaxExp);
    const std::string normExp = i32op("add", exp, std::to_string(127 - F.bias));
    const std::string normBits = i32op(
        "or", i32op("or", fsign, i32op("shl", normExp, "23")),
        i32op("shl", man, std::to_string(23 - F.M)));
    const std::string normRes = bitsf32(normBits);
    std::string topVal;
    if (F.nanBits != 0 && topNormal) {
      // e4m3: NaN only when man==allMant.
      const std::string isAll = i32cmp("eq", man, SallMant);
      const std::string asNan = bitsf32(i32op("or", fsign, "2143289344"));
      topVal = f32sel(isAll, asNan, normRes);
    } else if (F.nanBits != 0) {
      // e5m2: man==0 -> Inf else NaN.
      const std::string manZero = i32cmp("eq", man, "0");
      const std::string asInf = bitsf32(i32op("or", fsign, "2139095040"));
      const std::string asNan = bitsf32(i32op("or", fsign, "2143289344"));
      topVal = f32sel(manZero, asInf, asNan);
    } else {
      topVal = normRes;
    }
    const std::string r1 = f32sel(isTop, topVal, normRes);
    const std::string manNZ = i32cmp("ne", man, "0");
    const std::string isSubN = i1op("and", isZeroE, manNZ);
    const std::string r2 = f32sel(isSubN, subRes, r1);
    const std::string r3 = f32sel(isZero, zeroRes, r2);
    return r3;
  }

  bool emit_narrow_cvt(const rouge::PtxInstruction& ins, size_t idx,
                       const std::string& spec) {
    const auto& A = ins.args;
    const size_t dot = spec.find('.');
    const std::string dst = dot == std::string::npos ? spec : spec.substr(0, dot);
    const std::string src = dot == std::string::npos ? "" : spec.substr(dot + 1);
    const bool rz = ins.op.find(".rz.") != std::string::npos;
    const bool relu = ins.op.find(".relu") != std::string::npos;
    NarrowFmt F;
    // f32 -> packed x2 (d.b16 <- a upper, b lower).
    if ((dst == "e4m3x2" || dst == "e5m2x2" || dst == "e3m2x2" || dst == "e2m3x2" ||
         dst == "e2m1x2") &&
        src == "f32") {
      if (!narrow_fmt(dst, &F)) return fail("bad narrow type @" + std::to_string(idx));
      if (A.size() != 3)
        return fail("bad " + ins.op + " operands @ " + std::to_string(idx));
      const std::string fa = load_f32(A[1]);
      const std::string fb = load_f32(A[2]);
      const std::string ea = emit_fp_encode(fa, F, rz, relu);
      const std::string eb = emit_fp_encode(fb, F, rz, relu);
      // Pack into the container (e2m1 central, e3m2/e2m3 low-6, fp8 full).
      const auto packOne = [&](const std::string& field) {
        const std::string p = fresh("%pk");
        if (F.shift == 0 && F.width == 8) {
          line("  " + p + " = and i32 " + field + ", 255");
        } else {
          line("  " + p + " = shl i32 " +
               i32op("and", field, std::to_string((1u << F.width) - 1u)) + ", " +
               std::to_string(F.shift));
        }
        return p;
      };
      const std::string pa = packOne(ea);
      const std::string pb = packOne(eb);
      const std::string both = fresh("%pb");
      line("  " + both + " = or i32 " + i32op("shl", pa, "8") + ", " + pb);
      // The packed payload is 16 bits; a .b16 destination takes it truncated,
      // a wider register (nvcc uses .b32 for the packed pair) takes it as-is.
      if (is_reg(A[0]) && reg_bits(A[0]) == 16) {
        const std::string narrow = fresh("%nw");
        line("  " + narrow + " = trunc i32 " + both + " to i16");
        store_reg(A[0], narrow);
      } else {
        store_reg(A[0], both);
      }
      return true;
    }
    // packed x2 -> f16x2/bf16x2 (d = two halves).
    if ((dst == "f16x2" || dst == "bf16x2") &&
        (src == "e4m3x2" || src == "e5m2x2" || src == "e3m2x2" || src == "e2m3x2" ||
         src == "e2m1x2")) {
      if (!narrow_fmt(src, &F)) return fail("bad narrow type @" + std::to_string(idx));
      if (A.size() != 2)
        return fail("bad " + ins.op + " operands @ " + std::to_string(idx));
      std::string aval16 = operand(A[1]);
      // Source is a .b16 register holding two packed bytes.
      if (is_reg(A[1]) && reg_bits(A[1]) != 16) {
        const std::string t = fresh("%t");
        line("  " + t + " = trunc " + reg_type(A[1]) + " " + aval16 + " to i16");
        aval16 = t;
      }
      const std::string aval = fresh("%av");
      line("  " + aval + " = zext i16 " + aval16 + " to i32");
      const char* halfTy = dst == "f16x2" ? "half" : "bfloat";
      const auto cvtOne = [&](const std::string& byteSSA) {
        const std::string f = emit_fp_decode(byteSSA, F);
        // relu on the way out mirrors the interpreter (clamp negatives).
        std::string g = f;
        if (relu) {
          const std::string isNan = fresh("%qn");
          line("  " + isNan + " = fcmp uno float " + f + ", 0.0");
          const std::string isNeg = fresh("%qg");
          line("  " + isNeg + " = fcmp olt float " + f + ", 0.0");
          const std::string qnan = fresh("%qq");
          line("  " + qnan + " = bitcast i32 2139095040 to float");
          const std::string pos0 = fresh("%qz");
          line("  " + pos0 + " = bitcast i32 0 to float");
          const std::string g2 = fresh("%q2");
          line("  " + g2 + " = select i1 " + isNan + ", float " + qnan + ", float " + f);
          g = fresh("%q3");
          line("  " + g + " = select i1 " + isNeg + ", float " + pos0 + ", float " + g2);
        }
        const std::string h = fresh("%qh");
        line("  " + h + " = fptrunc float " + g + " to " + halfTy);
        const std::string b = fresh("%qb");
        line("  " + b + " = bitcast " + std::string(halfTy) + " " + h + " to i16");
        return b;
      };
      const std::string hi8 = i32op("and", i32op("lshr", aval, "8"), "255");
      const std::string lo8 = i32op("and", aval, "255");
      const std::string hi = cvtOne(hi8);
      const std::string lo = cvtOne(lo8);
      const std::string hi32 = fresh("%hc");
      line("  " + hi32 + " = zext i16 " + hi + " to i32");
      const std::string lo32 = fresh("%lc");
      line("  " + lo32 + " = zext i16 " + lo + " to i32");
      const std::string both = fresh("%cb");
      line("  " + both + " = or i32 " + i32op("shl", hi32, "16") + ", " + lo32);
      store_reg(A[0], both);
      return true;
    }
    return fail("unsupported narrow conversion '" + ins.op + "' @ " +
                std::to_string(idx) + " (x4/.rs/.scaled forms refuse loudly)");
  }

  bool emit_cvt(const rouge::PtxInstruction& ins, size_t idx) {
    const auto& A = ins.args;
    // Packed narrow-float forms take three operands (d, a, b); everything
    // else takes two. The arity is validated per form below.
    if (A.size() != 2) {
      const std::string& op = ins.op;
      const bool narrow3 =
          (op.find(".e4m3x2") != std::string::npos ||
           op.find(".e5m2x2") != std::string::npos ||
           op.find(".e3m2x2") != std::string::npos ||
           op.find(".e2m3x2") != std::string::npos ||
           op.find(".e2m1x2") != std::string::npos) &&
          op.find(".f32") != std::string::npos;
      if (!narrow3 || A.size() != 3)
        return fail("bad cvt operands @ " + std::to_string(idx));
    }
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
      const bool isMod = head == "sat" || head == "satfinite" || head == "relu" ||
                         head == "rn" || head == "rz" || head == "rzi" ||
                         head == "rp" || head == "ru" || head == "rm";
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
      // Narrow-float packed destinations carry mandatory .satfinite (clamp to
      // MAX_NORM) with .rn/.rz rounding; .relu is a plain clamp. x4, .rs and
      // .scaled companions refuse below, mirroring the interpreter.
      const auto isNarrow = [](const std::string& t) {
        return t == "e4m3x2" || t == "e5m2x2" || t == "e3m2x2" ||
               t == "e2m3x2" || t == "e2m1x2";
      };
      // Narrow-float packed types on either side (as destination with
      // mandatory .satfinite, or as source): .rn/.rz rounding plus
      // .satfinite/.relu clamps, all implemented below and in the interpreter.
      const bool narrowDst = isNarrow(dstTy);
      const bool narrowSrc = isNarrow(srcTy);
      const bool ok =
          mod.empty() || (mod == "rn" && isFloat(dstTy)) ||
          ((mod == "rz" || mod == "rzi") && !isFloat(dstTy) && isFloat(srcTy)) ||
          ((mod == "rn" || mod == "rz") && (narrowDst || narrowSrc)) ||
          ((mod == "satfinite" || mod == "relu") && (narrowDst || narrowSrc));
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
    // Narrow-float cvt forms (e4m3x2/...) are NOT bit-copies: they are routed
    // to emit_narrow_cvt below, which implements RNE+satfinite in integer IR.
    // NOTE: spec is already modifier-stripped here ("e4m3x2.f32"), so the
    // names match WITHOUT a leading dot (unlike ins.op, which keeps it).
    const bool narrowX2 =
        spec.find("e4m3x2") != std::string::npos ||
        spec.find("e5m2x2") != std::string::npos ||
        spec.find("e3m2x2") != std::string::npos ||
        spec.find("e2m3x2") != std::string::npos ||
        spec.find("e2m1x2") != std::string::npos;
    if (!narrowX2 && (spec.find(".f16x2") != std::string::npos ||
                       spec.find(".bf16x2") != std::string::npos)) {
      std::string val = operand(A[1]);
      const std::string dstTy = reg_type(A[0]);
      if (is_reg(A[1]) && reg_type(A[1]) != dstTy) {
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

    // Narrow-float packed conversions (e4m3x2/e5m2x2/e3m2x2/e2m3x2/e2m1x2):
    // same RNE+satfinite semantics as the interpreter (see rouge-ptx/ptx.cpp),
    // lowered to branchless integer IR below so every target (host, NVPTX,
    // AMD) executes them identically. x4/.rs/.scaled forms refuse loudly in
    // the gate above, like the interpreter.
    if (narrowX2) return emit_narrow_cvt(ins, idx, spec);

    // f16 / bf16 conversions: LLVM half / bfloat with fpext / fptrunc, which
    // round exactly like the interpreter's software conversions.
    const bool src16 = src == "f16" || src == "bf16";
    const bool dst16 = dst == "f16" || dst == "bf16";
    if (dst16 && is_reg(A[0]) && reg_bits(A[0]) != 16)
      return fail(ins.op + " needs a 16-bit destination (@ " +
                  std::to_string(idx) + ")");
    if (src16 && is_reg(A[1]) && reg_bits(A[1]) != 16)
      return fail(ins.op + " needs a 16-bit source (@ " + std::to_string(idx) + ")");
    if (dst16) {
      const char* irTy = dst == "f16" ? "half" : "bfloat";
      if (src == "f32" || src == "f64") {
        if (is_reg(A[1]) && reg_bits(A[1]) != 32)
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
    if (pair(dst, src) == "u64.u32") {
      const std::string z = fresh("%z");
      line("  " + z + " = zext i32 " + operand(A[1]) + " to i64");
      store_reg(A[0], z);
      return true;
    }
    // A SIGNED 32-bit source widens with a sign extension: zext would turn
    // -1 into 0xFFFFFFFF (a different value than the interpreter produces).
    if (pair(dst, src) == "s64.u32" || pair(dst, src) == "s64.s32") {
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
    // The source register's width must match the source type the mnemonic
    // names: "cvt.u64.u32" reads a 32-bit register. A mismatch used to reach
    // the lowering as a type error deep inside the emitted IR (clang then
    // complained about the .ll, not about the PTX).
    if ((src64i || src32i || src16i) && is_reg(A[1]) && reg_bits(A[1]) != 0 &&
        reg_bits(A[1]) != srcBits)
      return fail(ins.op + " expects a " + std::to_string(srcBits) +
                  "-bit source, but '" + A[1] + "' is " +
                  std::to_string(reg_bits(A[1])) + "-bit @ " +
                  std::to_string(idx));
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
    // The signedness comes from the TYPE (.s32 vs .u32), not from the
    // condition: "setp.lt.s32" is a SIGNED comparison. Picking the icmp
    // kind by the condition alone made every signed comparison unsigned —
    // the kernel then computed a different answer on the card than the
    // interpreter did (found by the differential fuzzer).
    const size_t dot = spec.find('.');
    const std::string ty = dot == std::string::npos ? "" : spec.substr(dot + 1);
    const bool is_signed = !ty.empty() && ty[0] == 's';
    const int bits = int_type_bits(ins.op);
    const std::string ity = bits == 64 ? "i64" : "i32";
    std::string ll;
    if (cond == "lt")
      ll = is_signed ? "slt" : "ult";
    else if (cond == "gt")
      ll = is_signed ? "sgt" : "ugt";
    else if (cond == "le")
      ll = is_signed ? "sle" : "ule";
    else if (cond == "ge")
      ll = is_signed ? "sge" : "uge";
    else if (cond == "eq")
      ll = "eq";
    else
      ll = "ne";
    const std::string x = operand(A[1]);
    const std::string y = operand(A[2]);
    const std::string c = fresh("%c");
    line("  " + c + " = icmp " + ll + " " + ity + " " + x + ", " + y);
    store_reg(A[0], c);
    return true;
  }

  // ---- target-specific address spaces -------------------------------------
  // Registers carry addresses as plain i64 (that is what PTX does) and are
  // turned into pointers at the point of use. WHICH typed pointer we must
  // produce depends on the target: AMDGPU and NVPTX number shared memory
  // differently, and a flat `ptr` makes the backend emit scalar ld/st through
  // an integer instead of real global_load/global_store.
  bool is_amdgpu() const { return target_.rfind("amdgcn", 0) == 0; }
  bool is_nvptx() const { return target_.rfind("nvptx", 0) == 0; }
  // CPU backends only accept the generic address space; emitting typed
  // pointers for a CPU target produces "expected ptr" verifier errors. The
  // typed form is strictly a GPU-target feature.
  bool gpu_addrspaces() const { return is_amdgpu() || is_nvptx(); }

  // global memory: addrspace(1) on both AMDGPU and NVPTX.
  std::string global_ptr_ty() const {
    return gpu_addrspaces() ? "ptr addrspace(1)" : "ptr";
  }
  // shared memory: addrspace(5) on AMDGPU, addrspace(3) on NVPTX.
  std::string shared_ptr_ty() const {
    if (!gpu_addrspaces()) return "ptr";
    return is_amdgpu() ? "ptr addrspace(5)" : "ptr addrspace(3)";
  }
  // Thread-local spill depots: NVPTX numbers its local window 5, so a
  // depot address through the generic space makes the backend guess (and
  // ptxas reject) a global access to a stack slot. Everywhere else the
  // generic pointer is already right (host flat memory, AMDGPU allocas).
  std::string local_ptr_ty() const {
    return is_nvptx() ? "ptr addrspace(5)" : "ptr";
  }

  // Turn an i64 host-visible address into a typed pointer.
  std::string inttoptr_as(const std::string& i64val, const std::string& ty) {
    const std::string p = fresh("%p");
    line("  " + p + " = inttoptr i64 " + i64val + " to " + ty);
    ptr_ty_[p] = ty;
    return p;
  }

  // True when a register holds the result of cvta.to.shared.
  bool is_shared_addr_reg(const std::string& reg) const {
    return shared_addr_regs_.count(reg) != 0;
  }
  // True when a register holds a .local depot address (mov from a .local
  // symbol or cvta.local). Such addresses must stay generic pointers:
  // defaulting them to the global space makes ptxas reject the backend's
  // st.global to a stack address (clang-emitted spill code caught this).
  bool is_local_addr_reg(const std::string& reg) const {
    return local_addr_regs_.count(reg) != 0;
  }

  std::string addr_of(const std::string& reg_tok, const std::string& space = "") {
    std::string a = load_reg(reg_tok);
    // A 32-bit address register is legal PTX (nvcc uses it for a demoted local
    // array: "mov.u32 %r, sym; add.s32 %r, %r, %off; st.shared.f32 [%r], %f"),
    // so widen it before turning it into a pointer.
    if (reg_type(reg_tok) != "i64") {
      const std::string w = fresh("%a");
      line("  " + w + " = zext " + reg_type(reg_tok) + " " + a + " to i64");
      a = w;
    }
    // A qualified opcode names its space authoritatively; a .local depot
    // address (or any provenance-marked shared address under a generic
    // opcode) keeps its own window instead of defaulting to global.
    if (space == "shared" || (space.empty() && is_shared_addr_reg(reg_tok)))
      return inttoptr_as(a, shared_ptr_ty());
    if (space.empty() && is_local_addr_reg(reg_tok))
      return inttoptr_as(a, local_ptr_ty());
    return inttoptr_as(a, global_ptr_ty());
  }

  // Read the block scratchpad base as i64. On the host it is a field of the
  // launch descriptor; on a GPU it is a module global that the compiler places
  // in the shared address space, so the kernel's own LDS window addresses it.
  std::string load_smem_base() {
    if (is_nvptx()) {
      const std::string a = fresh("%a");
      line("  " + a + " = ptrtoint ptr addrspace(3) @" + smem_global() +
           " to i64");
      return a;
    }
    const std::string gep = fresh("%lg");
    line("  " + gep + " = getelementptr inbounds i8, ptr %launch, i64 48");
    const std::string v = fresh("%v");
    line("  " + v + " = load i64, ptr " + gep + ", align 8");
    return v;
  }

  // The GPU block scratchpad. Its size is fixed at compile time by the
  // static .shared declarations, so a static global is enough; dynamic
  // (extern, unsized) variables are skipped here — they get a dedicated
  // extern global each (see emit_header). The address space is the one
  // NVPTX uses for .shared.
  std::string smem_global() {
    if (!smem_sym_.empty()) return smem_sym_;
    size_t bytes = 1;
    for (const auto& sv : fn_->sharedVars) {
      if (sv.dynamic) continue;
      bytes = std::max(bytes, static_cast<size_t>(sv.offset) +
                                      static_cast<size_t>(sv.elemBytes) *
                                          static_cast<size_t>(sv.count));
    }
    smem_sym_ = "__rouge_" + fn_->name + "_smem";
    // A global's alignment follows the initialiser, comma-separated. 16 bytes
    // is the widest shared access a kernel may perform.
    smem_decl_ = "@" + smem_sym_ + " = internal addrspace(3) global [" +
                 std::to_string(bytes) + " x i8] zeroinitializer, align 16";
    return smem_sym_;
  }

  // A .local spill-depot variable by operand name, or nullptr. Unlike
  // .shared (one copy per block), each thread owns its depot copy, lowered
  // to a plain function alloca.
  const rouge::PtxFunction::PtxLocalVar* find_local(
      const std::string& inner) const {
    const auto it = fn_->localIndex.find(inner);
    if (it == fn_->localIndex.end()) return nullptr;
    return &fn_->localVars[static_cast<size_t>(it->second)];
  }

  // Address of a .local depot variable as i64 (ptrtoint of its alloca).
  std::string local_addr(const std::string& name) {
    const std::string a = fresh("%la");
    line("  " + a + " = ptrtoint " + pt("%" + name) + " %" + name + " to i64");
    return a;
  }

  // If the operand names a .shared var (optionally with a +/-Const byte
  // suffix), return its offset and the constant part. Dynamic (extern,
  // unsized) vars resolve the same way; their base is separate (see
  // load_dsmem_base).
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

  // The .shared var named by an operand (with the same +/-Const suffix rule
  // as shared_symbol), or nullptr. Lets resolve_addr and the mov/cvta
  // address-takers pick the dynamic base for extern unsized variables.
  const rouge::PtxSharedVar* find_shared(const std::string& inner) const {
    for (const auto& sv : fn_->sharedVars) {
      if (inner == sv.name) return &sv;
      if (inner.size() > sv.name.size() &&
          inner.compare(0, sv.name.size(), sv.name) == 0) {
        const char c = inner[sv.name.size()];
        if (c == '+' || c == '-') return &sv;
      }
    }
    return nullptr;
  }

  // Dynamic .shared base as i64. On the host the dynamic region follows the
  // static scratchpad (sv.offset, laid out by the parser); on NVPTX it is a
  // dedicated extern addrspace(3) global whose storage the driver provides
  // from the launch's dynamic-shared allocation.
  std::string dsmem_sym(const std::string& name) const {
    return "__rouge_" + fn_->name + "_dsmem_" + name;
  }
  std::string load_dsmem_base(const rouge::PtxSharedVar& sv) {
    if (is_nvptx()) {
      const std::string a = fresh("%a");
      line("  " + a + " = ptrtoint ptr addrspace(3) @" + dsmem_sym(sv.name) +
           " to i64");
      return a;
    }
    const std::string base = load_smem_base();
    const std::string d = fresh("%db");
    line("  " + d + " = add i64 " + base + ", " + std::to_string(sv.offset));
    return d;
  }

  // If the operand (e.g. "[smem]" or "smem") names a .shared variable, return its
  // scratchpad base: the static window, or the dynamic base for extern unsized
  // variables.
  std::string shared_base_for(const std::string& inner) {
    const rouge::PtxSharedVar* sv = find_shared(inner);
    if (sv && sv->dynamic) return load_dsmem_base(*sv);
    return load_smem_base();
  }

  // Resolve a memory operand ("[%rd1]", "[smem]", "[smem+4]") to a `ptr` SSA:
  // shared symbols are scratchpad-relative, registers hold addresses as-is.
  std::string resolve_addr(const std::string& tok, const std::string& space = "") {
    const std::string inner = strip_brackets(tok);
    int varOffset = 0, constOff = 0;
    if (shared_symbol(inner, &varOffset, &constOff)) {
      const std::string base = shared_base_for(inner);
      const std::string off = fresh("%a");
      line("  " + off + " = add i64 " + base + ", " +
           std::to_string(varOffset + constOff));
      return inttoptr_as(off, shared_ptr_ty());
    }
    // Bare .local depot reference ("[__local_depot0]"): ptrtoint of its alloca.
    if (find_local(inner)) return inttoptr_as(local_addr(inner), pt("%" + inner));
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
      // A qualified opcode names its space authoritatively, so .shared
      // accesses resolve to the shared window even when the address sits in
      // an unmarked register; unqualified (generic) opcodes keep the
      // provenance fallback below (shared marks, then local marks, else the
      // global window).
      std::string ptrTy;
      if (space == "shared" || (space.empty() && is_shared_addr_reg(base)))
        ptrTy = shared_ptr_ty();
      else if (space.empty() && is_local_addr_reg(base))
        ptrTy = local_ptr_ty();
      else
        ptrTy = global_ptr_ty();
      return inttoptr_as(sum, ptrTy);
    }
    return addr_of(inner, space);
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
      const std::string addr = resolve_addr(list ? A[2] : A[1], op_space(ins.op));
      const std::string v = fresh("%v");
      line("  " + v + " = load i64, " + pt(addr) + " " + addr + ", align 8");
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
    std::string addr = resolve_addr(A[1], op_space(ins.op));
    if (ins.op.find(".f16x2") != std::string::npos ||
        ins.op.find(".bf16x2") != std::string::npos) {
      // f16x2/bf16x2: 32-bit payload (2×half) as i32
      const std::string v = fresh("%v");
      line("  " + v + " = load i32, " + pt(addr) + " " + addr + ", align 4");
      store_reg(A[0], v);
      return true;
    }
    if (is_8bit_ty(ins.op)) {
      // The destination is a normal (>= 32-bit) register, so the loaded byte is
      // zero-extended, except for .s8 which carries the sign (the interpreter's
      // rule for ld.global.s8).
      const std::string v = fresh("%v");
      line("  " + v + " = load i8, " + pt(addr) + " " + addr + ", align 1");
      const std::string e = fresh("%v");
      line("  " + e + " = " + (ins.op.rfind(".s8") != std::string::npos ? "sext" : "zext") +
           " i8 " + v + " to " + reg_type(A[0]));
      store_reg(A[0], e);
      return true;
    }
    if (is_16bit_ty(ins.op)) {
      const std::string v = fresh("%v");
      line("  " + v + " = load i16, " + pt(addr) + " " + addr + ", align 2");
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
      line("  " + v + " = load float, " + pt(addr) + " " + addr + ", align 4");
      store_f32(A[0], v);
      return true;
    }
    if (w64) {
      const std::string v = fresh("%v");
      line("  " + v + " = load i64, " + pt(addr) + " " + addr + ", align 8");
      store_reg(A[0], v);
      return true;
    }
    // Plain 32-bit load (u32/s32/b32, qualified or generic). A 64-bit
    // destination (the form clang's CUDA frontend emits, e.g. ld.s32 %rd4)
    // takes the word zero-extended — the interpreter's rule, so both paths
    // agree bit-for-bit.
    const std::string v = fresh("%v");
    line("  " + v + " = load i32, " + pt(addr) + " " + addr + ", align 4");
    store_reg(A[0], fit_reg(A[0], v, "i32"));
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
      const std::string addr = resolve_addr(A[0], op_space(ins.op));
      if (A.size() < 3) {  // plain 64-bit source register
        line("  store i64 " + operand(A[1]) + ", " + pt(addr) + " " + addr + ", align 8");
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
      line("  store i64 " + v + ", " + pt(addr) + " " + addr + ", align 8");
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
    std::string addr = resolve_addr(A[0], op_space(ins.op));
    if (ins.op.find(".f16x2") != std::string::npos ||
        ins.op.find(".bf16x2") != std::string::npos) {
      // f16x2/bf16x2: 32-bit payload as i32
      line("  store i32 " + operand(A[1]) + ", " + pt(addr) + " " + addr + ", align 4");
      return true;
    }
    if (is_8bit_ty(ins.op)) {
      // Any source register: the store takes the low byte.
      const std::string s = operand(A[1]);
      const std::string v = fresh("%v");
      line("  " + v + " = trunc " + reg_type(A[1]) + " " + s + " to i8");
      line("  store i8 " + v + ", " + pt(addr) + " " + addr + ", align 1");
      return true;
    }
    if (is_16bit_ty(ins.op)) {
      // A 32-bit source register is fine: the store takes its low halfword.
      // (Requiring a 16-bit source here used to reject valid PTX.)
      const std::string s = operand(A[1]);
      const std::string srcTy = operand_type(A[1], 16);
      if (srcTy == "i16") {
        line("  store i16 " + s + ", " + pt(addr) + " " + addr + ", align 2");
        return true;
      }
      const std::string v = fresh("%v");
      line("  " + v + " = trunc " + srcTy + " " + s + " to i16");
      line("  store i16 " + v + ", " + pt(addr) + " " + addr + ", align 2");
      return true;
    }
    if (f32) {
      const std::string bits = operand(A[1]);
      const std::string f = fresh("%f");
      line("  " + f + " = bitcast i32 " + bits + " to float");
      line("  store float " + f + ", " + pt(addr) + " " + addr + ", align 4");
      return true;
    }
    if (w64) {
      line("  store i64 " + operand(A[1]) + ", " + pt(addr) + " " + addr + ", align 8");
      return true;
    }
    line("  store i32 " + operand(A[1]) + ", " + pt(addr) + " " + addr + ", align 4");
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
      const std::string& raw = ins.args[0];
      // A branch to a label that is never defined must be a LOUD failure,
      // not an IR with a reference to a nonexistent block: the latter
      // produces IR that only the backend notices, and the error message
      // points at the backend, not at the PTX.
      if (!labels_.count(raw)) {
        fail("branch to an undefined label '" + raw + "'");
        return;
      }
      const std::string target = "b_" + raw;
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
                           std::string* error, const std::string& target) {
  if (fnIndex < 0 || static_cast<size_t>(fnIndex) >= prog.functions.size()) {
    if (error) *error = "kernel index out of range";
    return "";
  }
  const rouge::PtxFunction& fn = prog.functions[static_cast<size_t>(fnIndex)];
  LlvmGen gen(&fn, &prog, error, target);
  return gen.generate();
}

}  // namespace rougecomp