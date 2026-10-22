//===- PtxToRouge.cpp - the PTX front-end of the rouge dialect -*- C++ -*-===//
//
// The front-end: PTX -> a module of the `rouge` dialect. See
// rouge/Frontend/PtxToRouge.h for the model and for the reasons.
//
// THE RULE OF THIS FILE: everything that is not understood is REFUSED with a
// message that says what was not understood. A kernel that is translated
// into something slightly different is worse than a kernel that is refused:
// the first one computes the wrong thing silently.
//
//===----------------------------------------------------------------------===//

#include "rouge/Frontend/PtxToRouge.h"

#include "ptx.h"  // the shared PTX parser (the rouge namespace)

#include "rouge/IR/RougeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <cstdio>

using namespace mlir;

namespace rouge {
namespace {

// ---------------------------------------------------------------------------
// The address model
// ---------------------------------------------------------------------------

// What the analysis knows about the value in a register.
struct PtrValue {
  enum Kind { None, ParamPtr, Address, Offset } kind = None;
  // The value is the thread id (the index of the thread inside the block).
  // The vectoriser recognises "the thread id + a constant" as a unit-stride
  // access, so the front-end has to keep that shape instead of an integer
  // cast of it.
  bool isThreadId = false;
  // ParamPtr: the value of a kernel parameter (a pointer).
  int paramIndex = -1;
  // Address / Offset: base + index * scale, plus a chain of extra byte-offset
  // terms (the nvcc "demoted shared memory" form: mov.u32 %r, sym; add.s32
  // chains) and a constant byte offset (the "[reg+imm]" access form).
  int addrParam = -1;      // the kernel parameter the address is based on
  int addrShared = -1;     // the shared variable the address is based on
  std::string indexReg;    // the register with the element index ("" = none)
  int scale = 1;           // the byte scale of the index
  // Extra byte-offset terms: (register, scale). Each must satisfy
  // scale == elemBytes at the access, otherwise the access is refused.
  std::vector<std::pair<std::string, int>> extraTerms;
  // A constant byte offset (from "[reg+imm]" or from "add reg, reg, imm").
  long long constBytes = 0;
  bool isGlobal = true;    // global or shared
  // True when the (primary) index register is the thread id.
  // (kept above for clarity; field declared earlier)
};

// A memory access, recovered from the address arithmetic.
struct MemAccess {
  bool valid = false;
  // The index of the access is the thread id itself.
  bool indexIsThreadId = false;
  int paramIndex = -1;
  int sharedIndex = -1;
  std::string indexReg;
  int scale = 1;
  std::vector<std::pair<std::string, int>> extraTerms;
  long long constBytes = 0;
  bool isGlobal = true;
  Type elemType;
  int elemBytes = 4;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// The type of a value, from the PTX type suffix.
Type typeFromSuffix(StringRef suffix, MLIRContext *ctx) {
  // The MLIR-ish names (the hints use them) and the PTX ones.
  if (suffix == "i8") return IntegerType::get(ctx, 8);
  if (suffix == "i16") return IntegerType::get(ctx, 16);
  if (suffix == "i32") return IntegerType::get(ctx, 32);
  if (suffix == "i64") return IntegerType::get(ctx, 64);
  if (suffix == "b8" || suffix == "s8" || suffix == "u8") return IntegerType::get(ctx, 8);
  if (suffix == "b16" || suffix == "s16" || suffix == "u16") return IntegerType::get(ctx, 16);
  if (suffix == "b32" || suffix == "s32" || suffix == "u32") return IntegerType::get(ctx, 32);
  if (suffix == "b64" || suffix == "s64" || suffix == "u64") return IntegerType::get(ctx, 64);
  if (suffix == "f16") return Float16Type::get(ctx);
  if (suffix == "f32") return Float32Type::get(ctx);
  if (suffix == "f64") return Float64Type::get(ctx);
  if (suffix == "pred") return IntegerType::get(ctx, 1);
  return {};
}

int bytesOf(Type t) {
  if (auto it = dyn_cast<IntegerType>(t)) return std::max(1, (int)(it.getWidth() / 8));
  if (isa<Float16Type>(t)) return 2;
  if (isa<Float32Type>(t)) return 4;
  if (isa<Float64Type>(t)) return 8;
  return 4;
}

// The type suffix of an op: the part after the last '.' that looks like a
// type ("add.s32" -> "s32", "ld.global.f32" -> "f32").
StringRef typeSuffix(StringRef op) {
  size_t dot = op.rfind('.');
  if (dot == StringRef::npos) return {};
  return op.substr(dot + 1);
}

// The type suffix of the DESTINATION of a conversion ("cvt.u32.u64" -> "u32").
StringRef cvtDestSuffix(StringRef op) {
  // "cvt.<mode>.<dest>.<src>" or "cvt.<dest>.<src>".
  SmallVector<StringRef, 4> parts;
  op.split(parts, '.');
  if (parts.size() < 3) return {};
  // The mode (rn, rni, rz, ...) is a keyword, not a type.
  static const char *kModes[] = {"rn", "rni", "rz", "rnz", "rm", "rp", "rna"};
  if (parts.size() >= 4) {
    for (const char *m : kModes)
      if (parts[1] == m) return parts[2];
    return parts[2];
  }
  return parts[1];
}

bool isTypeSuffix(StringRef s) {
  return s == "b8" || s == "s8" || s == "u8" || s == "b16" || s == "s16" ||
         s == "u16" || s == "b32" || s == "s32" || s == "u32" || s == "b64" ||
         s == "s64" || s == "u64" || s == "f16" || s == "f32" || s == "f64" ||
         s == "pred";
}

// The token without the surrounding brackets ("[rd4]" -> "rd4").
std::string stripBrackets(StringRef tok) {
  std::string t = tok.str();
  if (t.size() >= 2 && t.front() == '[' && t.back() == ']')
    t = t.substr(1, t.size() - 2);
  return t;
}

// The register name without the '%' and without the predicate suffix
// ("r10|p1" -> "r10").
std::string regName(StringRef tok) {
  std::string t = tok.str();
  size_t bar = t.find('|');
  if (bar != std::string::npos) t = t.substr(0, bar);
  if (!t.empty() && t[0] == '%') t = t.substr(1);
  return t;
}

// The predicate of an instruction: "p1", "!p1" or "".
std::string predicateOf(const PtxInstruction &ins) {
  return ins.pred;
}

// True if the instruction WRITES a register (its first operand is the
// destination). The stores, the reductions and the branches do not: their
// first operand is an address or a label, and treating it as a destination
// would make the analysis think that the address register is redefined
// there (and then refuse a perfectly good kernel).
bool writesRegister(const PtxInstruction &ins) {
  const StringRef op = ins.op;
  if (op.starts_with("st.") || op.starts_with("red.")) return false;
  if (op == "bra" || op == "bra.uni" || op == "ret" || op == "exit") return false;
  if (op == "bar.sync" || op == "barrier") return false;
  if (op.starts_with("prefetch") || op.starts_with("membar") ||
      op.starts_with("fence."))
    return false;
  return true;
}

// ---------------------------------------------------------------------------
// The emitter
// ---------------------------------------------------------------------------

class PtxToRougeEmitter {
public:
  PtxToRougeEmitter(ModuleOp module, const PtxProgram &prog,
                    const PtxShapeHints &hints, std::string *error)
      : module_(module),
        prog_(prog),
        hints_(hints),
        error_(error),
        builder_(UnknownLoc::get(module.getContext()), module.getContext()),
        loc_(UnknownLoc::get(module.getContext())) {}

  LogicalResult run();

private:
  ModuleOp module_;
  const PtxProgram &prog_;
  const PtxShapeHints &hints_;
  std::string *error_;

  // --- the per-kernel state ---
  const PtxFunction *fn_ = nullptr;
  func::FuncOp func_;
  // MLIR 23: the OpBuilder::create overloads without a location need an
  // ImplicitLocOpBuilder (the plain OpBuilder wants the location every time).
  ImplicitLocOpBuilder builder_;
  Location loc_;
  llvm::StringMap<Value> regValue_;   // the register -> its SSA value
  llvm::StringMap<Type> regType_;     // the register -> its type
  llvm::StringMap<PtrValue> ptrValue_;  // the register -> what it holds
  llvm::StringMap<Value> paramValue_;  // the parameter index -> the value
  // The parameters: a pointer (a memref in the module) or a scalar.
  llvm::StringMap<bool> paramIsPointer_;
  llvm::StringMap<Type> paramScalarType_;
  std::vector<Value> sharedValue_;      // the shared variables -> the memrefs
  DenseMap<size_t, MemAccess> access_;  // the instruction -> its access
  std::set<size_t> skip_;               // the instructions not emitted

  // A loop-closing branch: the header -> the branch that jumps back to it.
  struct BackEdge {
    size_t index = 0;      // the branch itself
    std::string predicate; // "" for an unconditional back edge
  };
  std::map<size_t, BackEdge> loopBack_;

  // The refusal: the message goes into the error and the result is failure.
  LogicalResult fail(const std::string &msg) {
    if (error_) *error_ = msg;
    return failure();
  }

  // The type of a register (from the .reg declarations and the ops).
  Type typeOfReg(const std::string &name) {
    auto it = regType_.find(name);
    if (it != regType_.end()) return it->second;
    return {};
  }

  // The value of a register.
  Value valueOfReg(const std::string &name) {
    auto it = regValue_.find(name);
    if (it == regValue_.end()) return {};
    return it->second;
  }

  // A parameter register used as a pointer: the kind becomes Address, and
  // the parameter is marked as a pointer in the kernel's signature. Returns
  // true if the register was a parameter register.
  bool promoteParamToAddress(const std::string &reg);
  // The parameter registers that are the base of an address WITHOUT a cvta
  // (see analyseAddresses: only a register that reaches a dereference).
  llvm::StringSet<> pointerParams_;

  LogicalResult emitKernel(const PtxFunction &fn);
  LogicalResult inferTypes();
  LogicalResult analyseAddresses();
  LogicalResult buildSignature();
  LogicalResult emitBody();
  // Emit the instructions of [lo, hi) into the current insert block. A
  // backward branch closes a loop, a forward conditional branch is a guard
  // (an early exit) and everything else is one instruction. 'skips' marks the
  // instructions the ENCLOSING loop owns (its condition's setp and its exit
  // branch): they are not emitted where they stand, the loop builds them.
  LogicalResult emitRange(size_t lo, size_t hi, const std::set<size_t> &skips);
  // Emit the loop that starts at 'header' and is closed by the branch at
  // 'back' (exclusive).
  LogicalResult emitLoop(size_t header, size_t back);
  // Is the register still read after 'from' before anything redefines it?
  // (A read wins over a write: `add.s64 %rd, %rd, %r` reads before writing.)
  bool liveAfter(const std::string &reg, size_t from) const;
  LogicalResult emitInstruction(const PtxInstruction &ins, size_t index);
  // A constant of an integer type.
  Value constInt(Type t, int64_t v);
  // The value of an operand: a register, an immediate or a special register.
  Value operandValue(StringRef tok, Type type);
  // The index of an access, in elements.
  Value accessIndex(const MemAccess &acc);
  // The comparison of a setp.
  Value emitCompare(StringRef predOp, Value a, Value b, Type type);
};

LogicalResult PtxToRougeEmitter::run() {
  if (prog_.functions.empty()) return fail("the PTX program has no functions");

  for (const PtxFunction &fn : prog_.functions) {
    if (failed(emitKernel(fn))) return failure();
  }
  return success();
}

// The kernel's signature: the parameters become memrefs.
//
// The element type of a parameter comes from the accesses to it (the
// analysis), and the shape is dynamic unless the caller gave a hint: the
// PTX does not say how big the array is, and only the caller knows.
LogicalResult PtxToRougeEmitter::buildSignature() {
  SmallVector<Type> inputs;
  for (size_t i = 0; i < fn_->params.size(); ++i) {
    // A parameter that is not the base of an address is a scalar: the kernel
    // reads it with ld.param, and the PTX says of what type.
    if (!paramIsPointer_.count(std::to_string(i))) {
      auto sit = paramScalarType_.find(std::to_string(i));
      Type scalar = sit != paramScalarType_.end()
                        ? sit->second
                        : IntegerType::get(module_.getContext(), 64);
      if (!scalar) scalar = IntegerType::get(module_.getContext(), 64);
      inputs.push_back(scalar);
      continue;
    }
    Type elem = IntegerType::get(module_.getContext(), 32);
    int elemBytes = 4;
    // The element type of the parameter: the type of the accesses to it.
    for (const auto &kv : access_) {
      const MemAccess &acc = kv.second;
      if (acc.paramIndex != (int)i) continue;
      elem = acc.elemType;
      elemBytes = acc.elemBytes;
      break;
    }
    // A hint from the caller wins: it knows the real type.
    auto hintElem = hints_.elemTypes.find((int)i);
    if (hintElem != hints_.elemTypes.end()) {
      Type t = typeFromSuffix(hintElem->second, module_.getContext());
      if (!t) return fail("bad element type hint '" + hintElem->second + "'");
      elem = t;
      elemBytes = bytesOf(t);
    }
    // The shape: the hint, or dynamic.
    int64_t dim = ShapedType::kDynamic;
    auto hintShape = hints_.shapes.find((int)i);
    if (hintShape != hints_.shapes.end()) {
      // "64xf32" or "64" (the element type comes from the hint above).
      StringRef s = hintShape->second;
      size_t x = s.find('x');
      if (x == StringRef::npos) {
        dim = 0;
        if (s.getAsInteger(10, dim))
          return fail("bad shape hint '" + s.str() + "'");
      } else {
        dim = 0;
        if (s.substr(0, x).getAsInteger(10, dim))
          return fail("bad shape hint '" + s.str() + "'");
      }
    }
    // The memory space: the default one.
    Attribute space = MemRefType::get({1}, Float32Type::get(module_.getContext()))
                          .getMemorySpace();
    // MLIR 23: the layout comes before the memory space.
    inputs.push_back(MemRefType::get({dim}, elem, MemRefLayoutAttrInterface(), space));
    (void)elemBytes;
  }

  ImplicitLocOpBuilder moduleBuilder(loc_, module_.getContext());
  moduleBuilder.setInsertionPointToStart(module_.getBody());
  func_ = func::FuncOp::create(
      moduleBuilder, fn_->name,
      FunctionType::get(module_.getContext(), inputs, /*results=*/{}));
  Block *entry = func_.addEntryBlock();
  builder_ = ImplicitLocOpBuilder(loc_, module_.getContext());
  builder_.setInsertionPointToStart(entry);
  for (size_t i = 0; i < inputs.size(); ++i)
    paramValue_[std::to_string(i)] = entry->getArgument(i);
  return success();
}

// The type of every register: from the ops that write it.
LogicalResult PtxToRougeEmitter::inferTypes() {
  MLIRContext *ctx = module_.getContext();
  for (const PtxInstruction &ins : fn_->body) {
    if (ins.args.empty()) continue;
    const StringRef op = ins.op;
    // The instructions without a destination register.
    if (op == "bra" || op == "bra.uni" || op == "ret" || op == "exit" ||
        op == "bar.sync" || op == "barrier")
      continue;
    const std::string dst = regName(ins.args[0]);
    if (dst.empty()) continue;
    // A label, not a register.
    if (!ins.args[0].empty() && ins.args[0][0] == '$') continue;

    Type t;
    if (op == "setp" || op.starts_with("setp.")) {
      t = IntegerType::get(ctx, 1);
    } else if (op.starts_with("dp4a.")) {
      // The result of dp4a is 64 bits; the width comes from the register's
      // declaration (the PTX's own naming does not say it).
      auto it = fn_->regBits.find(dst);
      if (it != fn_->regBits.end()) {
        if (it->second == 64) t = IntegerType::get(ctx, 64);
        else if (it->second == 32) t = IntegerType::get(ctx, 32);
        else t = IntegerType::get(ctx, 64);
      }
    } else if (op.starts_with("mul.wide.") || op.starts_with("mad.wide.")) {
      // The suffix names the SOURCES (32-bit), the result is 64-bit. Taking
      // the suffix as the result type made a later redefinition of the same
      // register a 32-bit one, and an `add.s64` that used it became an
      // addi(i64, i32) — vadd.ptx found this.
      t = IntegerType::get(ctx, 64);
    } else if (op.starts_with("cvt.")) {
      t = typeFromSuffix(cvtDestSuffix(op), ctx);
    } else if (op.starts_with("mov.")) {
      // The destination's type: for a float move it is "f32", for an integer
      // move the suffix says so.
      t = typeFromSuffix(typeSuffix(op), ctx);
      if (!t) {
        // "mov.u32 %r, %tid.x" — the source decides.
        if (ins.args.size() > 1) {
          auto it = ptrValue_.find(regName(ins.args[1]));
          (void)it;
        }
      }
    } else if (op.starts_with("ld.") || op.starts_with("st.")) {
      // The type of a load: the suffix; a store's type comes from its
      // source, so it is not set here.
      t = typeFromSuffix(typeSuffix(op), ctx);
      if (op.starts_with("ld.")) {
        if (!t) return fail("cannot type the load '" + op.str() + "'");
        regType_[dst] = t;
      }
      continue;
    } else {
      t = typeFromSuffix(typeSuffix(op), ctx);
    }
    if (!t) {
      // The type may be known from the source register.
      if (ins.args.size() > 1) {
        Type src = typeOfReg(regName(ins.args[1]));
        if (src) t = src;
      }
    }
    if (!t) return fail("cannot type the register '" + dst + "' (" + op.str() + ")");
    regType_[dst] = t;
  }
  // The registers that are only read (the special ones) get their type from
  // the .reg declarations.
  for (const auto &kv : fn_->regBits) {
    if (regType_.count(kv.first)) continue;
    int bits = kv.second;
    if (bits == 1) regType_[kv.first] = IntegerType::get(ctx, 1);
    else if (bits == 16) regType_[kv.first] = IntegerType::get(ctx, 16);
    else if (bits == 32) regType_[kv.first] = IntegerType::get(ctx, 32);
    else if (bits == 64) regType_[kv.first] = IntegerType::get(ctx, 64);
  }
  return success();
}

// The address analysis: what each register holds (a pointer, an address, an
// offset) and what every memory instruction accesses.
bool PtxToRougeEmitter::promoteParamToAddress(const std::string &reg) {
  auto it = ptrValue_.find(reg);
  if (it == ptrValue_.end() || it->second.kind != PtrValue::ParamPtr)
    return false;
  it->second.kind = PtrValue::Address;
  it->second.addrParam = it->second.paramIndex;
  it->second.isGlobal = true;
  paramIsPointer_[std::to_string(it->second.paramIndex)] = true;
  return true;
}

LogicalResult PtxToRougeEmitter::analyseAddresses() {
  MLIRContext *ctx = module_.getContext();

  // Does the function have a back edge? A pointer ADVANCED inside a loop is
  // the one form the recovered "base + index" structure cannot express, see
  // the add branches below.
  bool hasLoop = false;
  for (size_t i = 0; i < fn_->body.size() && !hasLoop; ++i) {
    const PtxInstruction &ins = fn_->body[i];
    if (ins.op != "bra" || ins.args.empty()) continue;
    auto it = fn_->labels.find(ins.args[0]);
    if (it != fn_->labels.end() && (size_t)it->second <= i) hasLoop = true;
  }

  // The parameters: their index by name.
  llvm::StringMap<int> paramIndex;
  for (size_t i = 0; i < fn_->params.size(); ++i)
    paramIndex[fn_->params[i].name] = (int)i;

  // The shared variables: their index by name.
  llvm::StringMap<int> sharedIndex;
  for (size_t i = 0; i < fn_->sharedVars.size(); ++i)
    sharedIndex[fn_->sharedVars[i].name] = (int)i;

  // A parameter used as a pointer WITHOUT a cvta. nvcc normally emits
  // `cvta.to.global.u64` before touching a pointer parameter, but the ISA
  // does not require it: a generic address works for global memory, and
  // `atom.global.cas.b32 %r, [%p + %off]` is legal PTX (the atom_cas test
  // kernel is written that way). A .u64 parameter is promoted to a pointer
  // only when a register DERIVED FROM IT is dereferenced, so a scalar of the
  // same width is never mistaken for one.
  {
    llvm::StringSet<> deref;      // registers read as `[reg]`, `[reg + n]`
    llvm::StringSet<> param64;    // registers holding a .u64 parameter
    for (const PtxInstruction &ins : fn_->body) {
      if (StringRef(ins.op).starts_with("ld.param.") && ins.args.size() >= 2 &&
          typeSuffix(ins.op) == "u64")
        param64.insert(regName(ins.args[0]));
      for (const std::string &arg : ins.args) {
        if (arg.size() < 2 || arg.front() != '[') continue;
        std::string inner = stripBrackets(arg);
        const size_t cut = inner.find_first_of("+-");
        if (cut != std::string::npos) inner = inner.substr(0, cut);
        inner = regName(inner);
        if (!inner.empty()) deref.insert(inner);
      }
    }
    // Walk the additions backwards from the dereferences: the base of an
    // address is the register the chain starts at.
    llvm::StringSet<> addressLike = deref;
    bool changed = true;
    while (changed) {
      changed = false;
      for (const PtxInstruction &ins : fn_->body) {
        if (ins.op != "add.s64" && ins.op != "add.u64") continue;
        if (ins.args.size() < 3) continue;
        const std::string dst = regName(ins.args[0]);
        if (!addressLike.count(dst)) continue;
        for (int k = 1; k <= 2; ++k) {
          const std::string src = regName(ins.args[k]);
          if (param64.count(src)) pointerParams_.insert(src);
          if (!addressLike.count(src)) {
            addressLike.insert(src);
            changed = true;
          }
        }
      }
    }
  }

  // The constants a register holds, by definition point. nvcc materialises
  // the scale of an offset with `mov.u32 %rN, imm` and then uses the
  // REGISTER in `mul.wide %rd, %idx, %rN` (`gemm_tile.ptx` does exactly
  // that), so the analysis has to know the value of the register at the
  // point of the multiplication, not just literal immediates.
  struct ConstDef {
    size_t at;
    bool isConst;
    long long value;
  };
  std::map<std::string, std::vector<ConstDef>> constDefs;
  auto parseImm = [](const std::string &tok, long long &v) {
    if (tok.empty()) return false;
    if (!(std::isdigit((unsigned char)tok[0]) || tok[0] == '-' || tok[0] == '+'))
      return false;
    char *end = nullptr;
    v = strtoll(tok.c_str(), &end, 0);
    return end && *end == '\0';
  };
  for (size_t i = 0; i < fn_->body.size(); ++i) {
    const PtxInstruction &ins = fn_->body[i];
    if (ins.args.empty()) continue;
    // A store's first operand is an address, not a destination register.
    if (ins.args[0].empty() || ins.args[0][0] == '[' || ins.args[0][0] == '$')
      continue;
    const std::string d = regName(ins.args[0]);
    if (d.empty()) continue;
    const bool isMov = ins.op == "mov.u32" || ins.op == "mov.s32" ||
                       ins.op == "mov.b32" || ins.op == "mov.u64" ||
                       ins.op == "mov.s64" || ins.op == "mov.b64";
    long long v = 0;
    const bool isConst =
        isMov && ins.args.size() >= 2 && parseImm(ins.args[1], v);
    constDefs[d].push_back(ConstDef{i, isConst, v});
  }
  auto constantAt = [&](const std::string &r, size_t at, long long &out) {
    auto it = constDefs.find(r);
    if (it == constDefs.end()) return false;
    const std::vector<ConstDef> &defs = it->second;
    for (auto rit = defs.rbegin(); rit != defs.rend(); ++rit) {
      if (rit->at >= at) continue;
      if (!rit->isConst) return false;
      out = rit->value;
      return true;
    }
    return false;
  };

  for (size_t i = 0; i < fn_->body.size(); ++i) {
    const PtxInstruction &ins = fn_->body[i];
    const StringRef op = ins.op;
    if (ins.args.empty()) continue;

    // ld.param.u64 %rd, [name] -> the register holds a parameter's value.
    if (op.starts_with("ld.param.")) {
      if (ins.args.size() < 2) return fail("bad ld.param");
      std::string name = ins.args[1];
      if (name.size() > 1 && name.front() == '[' && name.back() == ']')
        name = name.substr(1, name.size() - 2);
      auto it = paramIndex.find(name);
      if (it == paramIndex.end()) return fail("unknown parameter '" + name + "'");
      PtrValue pv;
      pv.kind = PtrValue::ParamPtr;
      pv.paramIndex = it->second;
      ptrValue_[regName(ins.args[0])] = pv;
      // The parameter's own type: until a cvta makes it a pointer, it is a
      // scalar of the type the kernel reads it with.
      paramScalarType_[std::to_string(it->second)] =
          typeFromSuffix(typeSuffix(op), ctx);
      continue;
    }

    // mov.u32 %r, <shared symbol> -> a 32-bit shared address (nvcc's
    // "demoted" form: it never goes through cvta; the symbol IS the base).
    if (op.starts_with("mov.") && ins.args.size() >= 2) {
      std::string src = regName(ins.args[1]);
      // Strip brackets, if any.
      if (!src.empty() && src.front() == '[' && src.back() == ']')
        src = src.substr(1, src.size() - 2);
      auto sit = sharedIndex.find(src);
      if (sit != sharedIndex.end()) {
        PtrValue pv;
        pv.kind = PtrValue::Address;
        pv.isGlobal = false;
        pv.addrShared = sit->second;
        ptrValue_[regName(ins.args[0])] = pv;
        skip_.insert(i);
        continue;
      }
    }

    // cvta.to.global.u64 %rd, %src -> the register holds an address.
    if (op.starts_with("cvta.")) {
      if (ins.args.size() < 2) return fail("bad cvta");
      const std::string src = regName(ins.args[1]);
      PtrValue pv;
      pv.kind = PtrValue::Address;
      pv.isGlobal = op.find(".shared.") == StringRef::npos &&
                    op.find(".local.") == StringRef::npos;
      auto sit = sharedIndex.find(src);
      if (sit != sharedIndex.end()) {
        pv.addrShared = sit->second;
      } else {
        auto pit = ptrValue_.find(src);
        if (pit != ptrValue_.end() && pit->second.kind == PtrValue::ParamPtr) {
          pv.addrParam = pit->second.paramIndex;
          // The parameter is the base of an address: it is a pointer.
          paramIsPointer_[std::to_string(pit->second.paramIndex)] = true;
        } else {
          return fail("cvta from '" + src +
                      "' is not understood: only a kernel parameter or a "
                      "shared variable can be the base of an address");
        }
      }
      ptrValue_[regName(ins.args[0])] = pv;
      skip_.insert(i);
      continue;
    }

    // mul.wide.u32 %rd, %index, %scale -> an offset.
    if (op.starts_with("mul.wide.")) {
      if (ins.args.size() < 3) return fail("bad mul.wide");
      const std::string idx = regName(ins.args[1]);
      long scale = 1;
      const std::string &scaleTok = ins.args[2];
      long long scaleVal = 0;
      bool isImm = parseImm(scaleTok, scaleVal) ||
                   constantAt(regName(scaleTok), i, scaleVal);
      if (isImm) {
        scale = (long)scaleVal;
      } else {
        // The scale is not a constant: this is not an address offset but an
        // ordinary 64-bit value (a global index, for example). It is emitted
        // as it is, and it stays a candidate for an unscaled index.
        continue;
      }
      PtrValue pv;
      pv.kind = PtrValue::Offset;
      pv.indexReg = idx;
      pv.scale = (int)scale;
      auto it = ptrValue_.find(idx);
      if (it != ptrValue_.end() && it->second.kind == PtrValue::Offset)
        pv.constBytes = it->second.constBytes;
      ptrValue_[regName(ins.args[0])] = pv;
      skip_.insert(i);
      continue;
    }

    // mul.lo.* %r, %index, %scale -> a 32-bit offset (nvcc's demoted form
    // computes byte offsets with mul.lo.s32 instead of mul.wide.u64).
    if (op.starts_with("mul.lo.")) {
      if (ins.args.size() < 3) return fail("bad mul.lo");
      const std::string idx = regName(ins.args[1]);
      const std::string &scaleTok = ins.args[2];
      long long scaleVal = 0;
      bool isImm = parseImm(scaleTok, scaleVal) ||
                   constantAt(regName(scaleTok), i, scaleVal);
      if (isImm) {
        long scale = (long)scaleVal;
        PtrValue pv;
        pv.kind = PtrValue::Offset;
        pv.indexReg = idx;
        int baseScale = 1;
        auto it = ptrValue_.find(idx);
        long long baseConst = 0;
        if (it != ptrValue_.end() && it->second.kind == PtrValue::Offset) {
          baseScale = it->second.scale;
          baseConst = it->second.constBytes;
        }
        pv.scale = baseScale * (int)scale;
        // constBytes stays in the source's domain: mul.lo does not change
        // the unit (it scales the index, not the byte offset).
        pv.constBytes = baseConst;
        ptrValue_[regName(ins.args[0])] = pv;
        // Do NOT skip: mul.lo is also used for ordinary arithmetic, and the
        // recovered access uses the source register, not this result. Skipping
        // would leave regValue_ unset for the ordinary uses.
        continue;
      }
      // Non-constant scale: fall through to the ordinary-value path.
    }

    // add.s64 %rd, %a, %b -> the base plus the offset.
    if (op == "add.s64" || op == "add.u64") {
      if (ins.args.size() < 3) return fail("bad add.s64");
      const std::string a = regName(ins.args[1]);
      const std::string b = regName(ins.args[2]);
      // A pointer that MOVES. `add.s64 %p, %p, 16` (nvcc advances a tile
      // pointer on every iteration) cannot be folded into a constant byte
      // offset: the offset changes per iteration, and the recovered
      // memref[index] form would read the FIRST tile every time. The
      // front-end produced exactly that for int8_gemm.ptx — the card answered
      // CUDA_ERROR_ILLEGAL_ADDRESS, which is the good outcome; guessing is not.
      {
        auto dstIt = ptrValue_.find(regName(ins.args[0]));
        const bool moves = dstIt != ptrValue_.end() &&
                           dstIt->second.kind == PtrValue::Address &&
                           (!dstIt->second.indexReg.empty() ||
                            !dstIt->second.extraTerms.empty() ||
                            dstIt->second.constBytes != 0);
        if (hasLoop && moves && (regName(ins.args[0]) == a ||
                                 regName(ins.args[0]) == b))
          return fail("the address register '" + regName(ins.args[0]) +
                      "' is advanced inside a loop ('" + op.str() +
                      "'): a moving pointer is not expressible yet, and "
                      "folding the step into a constant byte offset would "
                      "read the first tile on every iteration");
      }
      // A parameter used as the base of an address without cvta.
      if (pointerParams_.count(a)) promoteParamToAddress(a);
      if (pointerParams_.count(b)) promoteParamToAddress(b);
      auto ai = ptrValue_.find(a);
      auto bi = ptrValue_.find(b);
      PtrValue pv;
      bool ok = false;
      if (ai != ptrValue_.end() && ai->second.kind == PtrValue::Address &&
          bi != ptrValue_.end() && bi->second.kind == PtrValue::Offset) {
        pv = ai->second;
        // Accumulate the offset term instead of replacing the previous one.
        const PtrValue &o = bi->second;
        if (pv.indexReg.empty() && pv.extraTerms.empty()) {
          pv.indexReg = o.indexReg;
          pv.scale = o.scale;
        } else if (!o.indexReg.empty()) {
          pv.extraTerms.push_back({o.indexReg, o.scale});
        }
        pv.extraTerms.insert(pv.extraTerms.end(), o.extraTerms.begin(),
                            o.extraTerms.end());
        pv.constBytes += o.constBytes;
        ok = true;
      } else if (bi != ptrValue_.end() && bi->second.kind == PtrValue::Address &&
                 ai != ptrValue_.end() && ai->second.kind == PtrValue::Offset) {
        pv = bi->second;
        const PtrValue &o = ai->second;
        if (pv.indexReg.empty() && pv.extraTerms.empty()) {
          pv.indexReg = o.indexReg;
          pv.scale = o.scale;
        } else if (!o.indexReg.empty()) {
          pv.extraTerms.push_back({o.indexReg, o.scale});
        }
        pv.extraTerms.insert(pv.extraTerms.end(), o.extraTerms.begin(),
                            o.extraTerms.end());
        pv.constBytes += o.constBytes;
        ok = true;
      } else if (ai != ptrValue_.end() && ai->second.kind == PtrValue::Address) {
        // Address + a constant immediate.
        const std::string &tok = ins.args[2];
        bool numeric = !tok.empty() &&
                       (std::isdigit((unsigned char)tok[0]) || tok[0] == '-');
        if (numeric) {
          pv = ai->second;
          char *end = nullptr;
          long long imm = strtoll(tok.c_str(), &end, 10);
          if (end && *end == '\0') {
            pv.constBytes += imm;
            ok = true;
          }
        }
      } else if (bi != ptrValue_.end() && bi->second.kind == PtrValue::Address) {
        const std::string &tok = ins.args[1];
        bool numeric = !tok.empty() &&
                       (std::isdigit((unsigned char)tok[0]) || tok[0] == '-');
        if (numeric) {
          pv = bi->second;
          char *end = nullptr;
          long long imm = strtoll(tok.c_str(), &end, 10);
          if (end && *end == '\0') {
            pv.constBytes += imm;
            ok = true;
          }
        }
      }
      if (!ok) {
        // A plain 64-bit addition: an ordinary value.
        continue;
      }
      ptrValue_[regName(ins.args[0])] = pv;
      skip_.insert(i);
      continue;
    }

    // add.s32 / add.u32 %r, %a, %b -> base plus offset (nvcc's 32-bit
    // demoted form). Mirrors the 64-bit case, including the accumulation of
    // extra byte-offset terms and constant byte offsets.
    if (op == "add.s32" || op == "add.u32") {
      if (ins.args.size() < 3) return fail("bad add.s32");
      const std::string a = regName(ins.args[1]);
      const std::string b = regName(ins.args[2]);
      {
        auto dstIt = ptrValue_.find(regName(ins.args[0]));
        const bool moves = dstIt != ptrValue_.end() &&
                           dstIt->second.kind == PtrValue::Address &&
                           (!dstIt->second.indexReg.empty() ||
                            !dstIt->second.extraTerms.empty() ||
                            dstIt->second.constBytes != 0);
        if (hasLoop && moves && (regName(ins.args[0]) == a ||
                                 regName(ins.args[0]) == b))
          return fail("the address register '" + regName(ins.args[0]) +
                      "' is advanced inside a loop ('" + op.str() +
                      "'): a moving pointer is not expressible yet, and "
                      "folding the step into a constant byte offset would "
                      "read the first tile on every iteration");
      }
      auto ai = ptrValue_.find(a);
      auto bi = ptrValue_.find(b);
      PtrValue pv;
      bool ok = false;
      // Address + Offset
      if (ai != ptrValue_.end() && ai->second.kind == PtrValue::Address &&
          bi != ptrValue_.end() && bi->second.kind == PtrValue::Offset) {
        pv = ai->second;
        const PtrValue &o = bi->second;
        if (pv.indexReg.empty() && pv.extraTerms.empty()) {
          pv.indexReg = o.indexReg;
          pv.scale = o.scale;
        } else if (!o.indexReg.empty()) {
          pv.extraTerms.push_back({o.indexReg, o.scale});
        }
        pv.extraTerms.insert(pv.extraTerms.end(), o.extraTerms.begin(),
                            o.extraTerms.end());
        pv.constBytes += o.constBytes;
        ok = true;
      } else if (bi != ptrValue_.end() && bi->second.kind == PtrValue::Address &&
                 ai != ptrValue_.end() && ai->second.kind == PtrValue::Offset) {
        pv = bi->second;
        const PtrValue &o = ai->second;
        if (pv.indexReg.empty() && pv.extraTerms.empty()) {
          pv.indexReg = o.indexReg;
          pv.scale = o.scale;
        } else if (!o.indexReg.empty()) {
          pv.extraTerms.push_back({o.indexReg, o.scale});
        }
        pv.extraTerms.insert(pv.extraTerms.end(), o.extraTerms.begin(),
                            o.extraTerms.end());
        pv.constBytes += o.constBytes;
        ok = true;
      } else if (ai != ptrValue_.end() && ai->second.kind == PtrValue::Address) {
        const std::string &tok = ins.args[2];
        bool numeric = !tok.empty() &&
                       (std::isdigit((unsigned char)tok[0]) || tok[0] == '-');
        if (numeric) {
          pv = ai->second;
          char *end = nullptr;
          long long imm = strtoll(tok.c_str(), &end, 10);
          if (end && *end == '\0') {
            pv.constBytes += imm;
            ok = true;
          }
        }
      } else if (bi != ptrValue_.end() && bi->second.kind == PtrValue::Address) {
        const std::string &tok = ins.args[1];
        bool numeric = !tok.empty() &&
                       (std::isdigit((unsigned char)tok[0]) || tok[0] == '-');
        if (numeric) {
          pv = bi->second;
          char *end = nullptr;
          long long imm = strtoll(tok.c_str(), &end, 10);
          if (end && *end == '\0') {
            pv.constBytes += imm;
            ok = true;
          }
        }
      }
      if (!ok) {
        // A plain 32-bit addition: an ordinary value.
        continue;
      }
      pv.kind = (ai != ptrValue_.end() && (ai->second.kind == PtrValue::Address ||
                  (bi != ptrValue_.end() && bi->second.kind == PtrValue::Address)))
                ? PtrValue::Address : PtrValue::Offset;
      // Preserve address-ness: if either side was an Address, the result is.
      if ((ai != ptrValue_.end() && ai->second.kind == PtrValue::Address) ||
          (bi != ptrValue_.end() && bi->second.kind == PtrValue::Address))
        pv.kind = PtrValue::Address;
      else
        pv.kind = PtrValue::Offset;
      // Preserve addrParam/addrShared/isGlobal from the address side if present.
      if (ai != ptrValue_.end() && ai->second.kind == PtrValue::Address) {
        pv.addrParam = ai->second.addrParam;
        pv.addrShared = ai->second.addrShared;
        pv.isGlobal = ai->second.isGlobal;
      } else if (bi != ptrValue_.end() && bi->second.kind == PtrValue::Address) {
        pv.addrParam = bi->second.addrParam;
        pv.addrShared = bi->second.addrShared;
        pv.isGlobal = bi->second.isGlobal;
      }
      ptrValue_[regName(ins.args[0])] = pv;
      skip_.insert(i);
      continue;
    }

    // A conversion: the value is preserved (modulo a truncation, which the
    // PTX allows), so the index candidate survives the conversion.
    if (op.starts_with("cvt.")) {
      if (ins.args.size() < 2) return fail("bad cvt");
      const std::string src = regName(ins.args[1]);
      auto it = ptrValue_.find(src);
      PtrValue pv;
      pv.kind = PtrValue::Offset;
      pv.indexReg = regName(ins.args[0]);
      pv.scale = 1;
      if (it != ptrValue_.end() && it->second.kind == PtrValue::Offset) {
        pv.scale = it->second.scale;
        pv.extraTerms = it->second.extraTerms;
        pv.constBytes = it->second.constBytes;
        pv.isThreadId = it->second.isThreadId;
      } else if (it != ptrValue_.end() && it->second.kind == PtrValue::Address) {
        // A conversion of an address: keep it as an address (the base is
        // preserved; the index candidate is the destination register).
        pv.kind = PtrValue::Address;
        pv.addrParam = it->second.addrParam;
        pv.addrShared = it->second.addrShared;
        pv.isGlobal = it->second.isGlobal;
        pv.indexReg = it->second.indexReg;
        pv.scale = it->second.scale;
        pv.extraTerms = it->second.extraTerms;
        pv.constBytes = it->second.constBytes;
        pv.isThreadId = it->second.isThreadId;
      }
      ptrValue_[regName(ins.args[0])] = pv;
      continue;
    }

    // A move of the thread id (x dimension only): the value IS the thread id,
    // and the vectoriser recognises it as such (a unit-stride access).
    if (op.starts_with("mov.") && ins.args.size() > 1) {
      std::string src = regName(ins.args[1]);
      bool isTidX = (src == "tid" || src == "tid.x");
      if (isTidX) {
        PtrValue pv;
        pv.kind = PtrValue::Offset;
        pv.indexReg = regName(ins.args[0]);
        pv.scale = 1;
        pv.isThreadId = true;
        ptrValue_[regName(ins.args[0])] = pv;
        continue;
      }
    }

    // Every other value is a CANDIDATE for an element index with the scale
    // one: the PTX often computes the byte offset directly (the element is
    // one byte wide — the int8 case, for example). The scale is CHECKED at
    // the access: it must be exactly the element size, so a wrong guess is
    // refused rather than silently believed.
    {
      const std::string dstReg = regName(ins.args[0]);
      if (!dstReg.empty() && !ptrValue_.count(dstReg)) {
        PtrValue pv;
        pv.kind = PtrValue::Offset;
        pv.indexReg = dstReg;
        pv.scale = 1;
        ptrValue_[dstReg] = pv;
      }
    }

    // A memory access: the address register must be understood.
    if (op.starts_with("ld.") && op.find(".param.") == StringRef::npos) {
      if (ins.args.size() < 2) return fail("bad load");
      // The access may carry a constant byte offset: "[reg+imm]" or "[reg-imm]".
      std::string bracketed = stripBrackets(ins.args[1]);
      std::string addr = bracketed;
      long long imm = 0;
      bool hasImm = false;
      size_t pos = bracketed.find('+');
      size_t neg = bracketed.find('-', 1);
      if (pos != std::string::npos) {
        addr = bracketed.substr(0, pos);
        std::string tok = bracketed.substr(pos + 1);
        // trim
        while (!tok.empty() && std::isspace((unsigned char)tok.front())) tok.erase(tok.begin());
        while (!tok.empty() && std::isspace((unsigned char)tok.back())) tok.pop_back();
        while (!addr.empty() && std::isspace((unsigned char)addr.back())) addr.pop_back();
        char *end = nullptr;
        long long v = strtoll(tok.c_str(), &end, 10);
        if (end && *end == '\0' && !tok.empty()) { imm = v; hasImm = true; }
      } else if (neg != std::string::npos) {
        addr = bracketed.substr(0, neg);
        std::string tok = bracketed.substr(neg + 1);
        while (!tok.empty() && std::isspace((unsigned char)tok.front())) tok.erase(tok.begin());
        while (!tok.empty() && std::isspace((unsigned char)tok.back())) tok.pop_back();
        while (!addr.empty() && std::isspace((unsigned char)addr.back())) addr.pop_back();
        char *end = nullptr;
        long long v = strtoll(tok.c_str(), &end, 10);
        if (end && *end == '\0' && !tok.empty()) { imm = -v; hasImm = true; }
      }
      // trim addr
      while (!addr.empty() && std::isspace((unsigned char)addr.front())) addr.erase(addr.begin());
      while (!addr.empty() && std::isspace((unsigned char)addr.back())) addr.pop_back();
      const std::string reg = regName(addr);
      // A parameter dereferenced directly (`ld.global [%rd1]`) is a pointer:
      // nothing else can be dereferenced.
      promoteParamToAddress(reg);
      auto it = ptrValue_.find(reg);
      if (it == ptrValue_.end() || it->second.kind != PtrValue::Address)
        return fail("the address of the load '" + ins.args[1] +
                    "' is not understood: the front-end recovers only "
                    "param/global + index*element size");
      MemAccess acc;
      acc.valid = true;
      acc.paramIndex = it->second.addrParam;
      acc.sharedIndex = it->second.addrShared;
      acc.indexReg = it->second.indexReg;
      acc.scale = it->second.scale;
      acc.extraTerms = it->second.extraTerms;
      acc.constBytes = it->second.constBytes + (hasImm ? imm : 0);
      acc.isGlobal = it->second.isGlobal;
      acc.elemType = typeFromSuffix(typeSuffix(op), ctx);
      if (!acc.elemType)
        return fail("the element type of the load '" + op.str() + "' is not known");
      acc.elemBytes = bytesOf(acc.elemType);
      if (!acc.indexReg.empty()) {
        auto it2 = ptrValue_.find(acc.indexReg);
        if (it2 != ptrValue_.end() && it2->second.isThreadId)
          acc.indexIsThreadId = true;
      }
      access_[i] = acc;
      continue;
    }
    if (op.starts_with("st.") && op.find(".param.") == StringRef::npos) {
      if (ins.args.size() < 2) return fail("bad store");
      std::string bracketed = stripBrackets(ins.args[0]);
      std::string addr = bracketed;
      long long imm = 0;
      bool hasImm = false;
      size_t pos = bracketed.find('+');
      size_t neg = bracketed.find('-', 1);
      if (pos != std::string::npos) {
        addr = bracketed.substr(0, pos);
        std::string tok = bracketed.substr(pos + 1);
        while (!tok.empty() && std::isspace((unsigned char)tok.front())) tok.erase(tok.begin());
        while (!tok.empty() && std::isspace((unsigned char)tok.back())) tok.pop_back();
        while (!addr.empty() && std::isspace((unsigned char)addr.back())) addr.pop_back();
        char *end = nullptr;
        long long v = strtoll(tok.c_str(), &end, 10);
        if (end && *end == '\0' && !tok.empty()) { imm = v; hasImm = true; }
      } else if (neg != std::string::npos) {
        addr = bracketed.substr(0, neg);
        std::string tok = bracketed.substr(neg + 1);
        while (!tok.empty() && std::isspace((unsigned char)tok.front())) tok.erase(tok.begin());
        while (!tok.empty() && std::isspace((unsigned char)tok.back())) tok.pop_back();
        while (!addr.empty() && std::isspace((unsigned char)addr.back())) addr.pop_back();
        char *end = nullptr;
        long long v = strtoll(tok.c_str(), &end, 10);
        if (end && *end == '\0' && !tok.empty()) { imm = -v; hasImm = true; }
      }
      while (!addr.empty() && std::isspace((unsigned char)addr.front())) addr.erase(addr.begin());
      while (!addr.empty() && std::isspace((unsigned char)addr.back())) addr.pop_back();
      const std::string reg = regName(addr);
      promoteParamToAddress(reg);
      auto it = ptrValue_.find(reg);
      if (it == ptrValue_.end() || it->second.kind != PtrValue::Address)
        return fail("the address of the store '" + ins.args[0] +
                    "' is not understood: the front-end recovers only "
                    "param/global + index*element size");
      MemAccess acc;
      acc.valid = true;
      acc.paramIndex = it->second.addrParam;
      acc.sharedIndex = it->second.addrShared;
      acc.indexReg = it->second.indexReg;
      acc.scale = it->second.scale;
      acc.extraTerms = it->second.extraTerms;
      acc.constBytes = it->second.constBytes + (hasImm ? imm : 0);
      acc.isGlobal = it->second.isGlobal;
      acc.elemType = typeFromSuffix(typeSuffix(op), ctx);
      if (!acc.elemType)
        return fail("the element type of the store '" + op.str() + "' is not known");
      acc.elemBytes = bytesOf(acc.elemType);
      if (!acc.indexReg.empty()) {
        auto it2 = ptrValue_.find(acc.indexReg);
        if (it2 != ptrValue_.end() && it2->second.isThreadId)
          acc.indexIsThreadId = true;
      }
      access_[i] = acc;
      continue;
    }
    // Atomics: atom.* / red.* — the same address recovery as a load/store,
    // plus a value operand (and, for cas, a compare and a new value).
    if ((op.starts_with("atom.") || op.starts_with("red.")) &&
        op.find(".param.") == StringRef::npos) {
      const bool hasDst = op.starts_with("atom.");
      const size_t addrArg = hasDst ? 1 : 0;
      if (ins.args.size() < addrArg + 1) return fail("bad atomic");
      std::string bracketed = stripBrackets(ins.args[addrArg]);
      std::string addr = bracketed;
      long long imm = 0;
      bool hasImm = false;
      size_t pos = bracketed.find('+');
      size_t neg = bracketed.find('-', 1);
      if (pos != std::string::npos) {
        addr = bracketed.substr(0, pos);
        std::string tok = bracketed.substr(pos + 1);
        while (!tok.empty() && std::isspace((unsigned char)tok.front())) tok.erase(tok.begin());
        while (!tok.empty() && std::isspace((unsigned char)tok.back())) tok.pop_back();
        while (!addr.empty() && std::isspace((unsigned char)addr.back())) addr.pop_back();
        char *end = nullptr;
        long long v = strtoll(tok.c_str(), &end, 10);
        if (end && *end == '\\0' && !tok.empty()) { imm = v; hasImm = true; }
      } else if (neg != std::string::npos) {
        addr = bracketed.substr(0, neg);
        std::string tok = bracketed.substr(neg + 1);
        while (!tok.empty() && std::isspace((unsigned char)tok.front())) tok.erase(tok.begin());
        while (!tok.empty() && std::isspace((unsigned char)tok.back())) tok.pop_back();
        while (!addr.empty() && std::isspace((unsigned char)addr.back())) addr.pop_back();
        char *end = nullptr;
        long long v = strtoll(tok.c_str(), &end, 10);
        if (end && *end == '\\0' && !tok.empty()) { imm = -v; hasImm = true; }
      }
      while (!addr.empty() && std::isspace((unsigned char)addr.front())) addr.erase(addr.begin());
      while (!addr.empty() && std::isspace((unsigned char)addr.back())) addr.pop_back();
      const std::string reg = regName(addr);
      promoteParamToAddress(reg);
      auto it = ptrValue_.find(reg);
      if (it == ptrValue_.end() || it->second.kind != PtrValue::Address)
        return fail("the address of the atomic '" + ins.args[addrArg] +
                    "' is not understood");
      MemAccess acc;
      acc.valid = true;
      acc.paramIndex = it->second.addrParam;
      acc.sharedIndex = it->second.addrShared;
      acc.indexReg = it->second.indexReg;
      acc.scale = it->second.scale;
      acc.extraTerms = it->second.extraTerms;
      acc.constBytes = it->second.constBytes + (hasImm ? imm : 0);
      acc.isGlobal = it->second.isGlobal;
      acc.elemType = typeFromSuffix(typeSuffix(op), ctx);
      if (!acc.elemType)
        return fail("the element type of the atomic '" + op.str() + "' is not known");
      acc.elemBytes = bytesOf(acc.elemType);
      if (!acc.indexReg.empty()) {
        auto it2 = ptrValue_.find(acc.indexReg);
        if (it2 != ptrValue_.end() && it2->second.isThreadId)
          acc.indexIsThreadId = true;
      }
      access_[i] = acc;
      continue;
    }
  }
  return success();
}

// A constant of an integer type.
Value PtxToRougeEmitter::constInt(Type t, int64_t v) {
  auto intTy = dyn_cast<IntegerType>(t);
  if (!intTy) return {};
  return builder_.create<arith::ConstantOp>(
      loc_, builder_.getIntegerAttr(intTy, v));
}

// The value of an operand: a register, an immediate or a special register.
Value PtxToRougeEmitter::operandValue(StringRef tok, Type type) {
  if (tok.empty()) return {};
  // A number? A leading '-' is part of the literal: `mov.s32 %r, -16` is
  // what nvcc emits for a negative constant, and refusing it would refuse a
  // kernel for no good reason. A lone sign is not a number (see below).
  bool numeric = true;
  for (char c : tok)
    if (!std::isdigit((unsigned char)c) && c != '-' && c != '+') numeric = false;
  if (numeric && !tok.empty() &&
      tok.find_first_of("0123456789") != StringRef::npos) {
    // The sign is handled here: StringRef::getAsInteger does not take one.
    const bool negative = tok[0] == '-';
    StringRef digits = tok;
    if (negative || tok[0] == '+') digits = digits.drop_front();
    APInt value(64, 0);
    if (!digits.getAsInteger(10, value)) {
      if (negative) value = -value;
      if (auto intTy = dyn_cast<IntegerType>(type))
        return builder_.create<arith::ConstantOp>(
            builder_.getIntegerAttr(intTy, value.trunc(intTy.getWidth())));
      // A float register: `mov.u16 %h, 0` is how nvcc zeroes an f16
      // accumulator (the .b16 register is viewed as f16 by the arithmetic),
      // so the constant is materialised in the register's own type.
      if (isa<FloatType>(type))
        return builder_.create<arith::ConstantOp>(builder_.getFloatAttr(
            type, (double)value.getSExtValue()));
    }
    return {};
  }
  if (tok.starts_with("0x")) {
    APInt value(64, 0);
    if (!tok.drop_front(2).getAsInteger(16, value)) {
      if (auto intTy = dyn_cast<IntegerType>(type))
        return builder_.create<arith::ConstantOp>(
            builder_.getIntegerAttr(intTy, value.trunc(intTy.getWidth())));
    }
    return {};
  }
  // A special register (%tid.x/.y/.z, %ctaid.x/.y/.z, ...).
  std::string name = regName(tok);
  rouge::Dimension dim = rouge::Dimension::x;
  if (name.size() > 2 && name[name.size()-2] == '.') {
    char suf = name.back();
    if (suf == 'x' || suf == 'y' || suf == 'z') {
      if (suf == 'y') dim = rouge::Dimension::y;
      else if (suf == 'z') dim = rouge::Dimension::z;
      name = name.substr(0, name.size() - 2);
    }
  }
  if (name == "tid") {
    auto op = builder_.create<rouge::ThreadIdOp>(
        loc_, builder_.getIndexType(),
        rouge::DimensionAttr::get(module_.getContext(), dim));
    return op.getResult();
  }
  if (name == "ctaid") {
    auto op = builder_.create<rouge::BlockIdOp>(
        loc_, builder_.getIndexType(),
        rouge::DimensionAttr::get(module_.getContext(), dim));
    return op.getResult();
  }
  if (name == "ntid") {
    auto op = builder_.create<rouge::BlockDimOp>(
        loc_, builder_.getIndexType(),
        rouge::DimensionAttr::get(module_.getContext(), dim));
    return op.getResult();
  }
  if (name == "nctaid") {
    auto op = builder_.create<rouge::GridDimOp>(
        loc_, builder_.getIndexType(),
        rouge::DimensionAttr::get(module_.getContext(), dim));
    return op.getResult();
  }
  // An ordinary register.
  return valueOfReg(name);
}

// The index of an access, in elements.
Value PtxToRougeEmitter::accessIndex(const MemAccess &acc) {
  MLIRContext *ctx = module_.getContext();
  const int eb = acc.elemBytes;
  // Thread id fast path: keep the bare thread id (or thread id + a constant)
  // so the vectoriser recognises a unit-stride access.
  if (acc.indexIsThreadId && acc.extraTerms.empty()) {
    if (acc.scale != eb) return {};
    auto tid = builder_.create<rouge::ThreadIdOp>(
        loc_, builder_.getIndexType(),
        rouge::DimensionAttr::get(ctx, rouge::Dimension::x));
    Value idx = tid.getResult();
    if (acc.constBytes != 0) {
      if (acc.constBytes % eb != 0) return {};
      long long c = acc.constBytes / eb;
      Value cv = builder_.create<arith::ConstantIndexOp>(loc_, c);
      idx = builder_.create<arith::AddIOp>(idx, cv).getResult();
    }
    return idx;
  }

  // A term of the address: the register's index in ELEMENTS. A byte scale
  // that is a whole number of elements is a stride, not an error: the block
  // index of a tiled kernel is exactly that (`gemm_tile.ptx`: the block
  // offset is blockId*512 bytes of f16, i.e. blockId*256 elements, and the
  // thread's own offset is tlinear*2 bytes).
  auto toIndex = [&](Value raw) -> Value {
    if (!raw) return {};
    if (raw.getType().isIndex()) return raw;
    if (auto intTy = dyn_cast<IntegerType>(raw.getType())) {
      if (intTy.getWidth() == 1) return {};
      return builder_.create<arith::IndexCastOp>(builder_.getIndexType(), raw)
          .getResult();
    }
    return {};
  };
  auto scaled = [&](Value raw, int scale) -> Value {
    Value v = toIndex(raw);
    if (!v) return {};
    if (scale == eb) return v;
    if (scale % eb != 0) return {};
    Value c = builder_.create<arith::ConstantIndexOp>(loc_, scale / eb);
    return builder_.create<arith::MulIOp>(v, c).getResult();
  };

  // Build the primary term.
  Value idx;
  if (acc.indexIsThreadId) {
    auto tid = builder_.create<rouge::ThreadIdOp>(
        loc_, builder_.getIndexType(),
        rouge::DimensionAttr::get(ctx, rouge::Dimension::x));
    idx = scaled(tid.getResult(), acc.scale);
  } else if (!acc.indexReg.empty()) {
    idx = scaled(valueOfReg(acc.indexReg), acc.scale);
  } else {
    idx = builder_.create<arith::ConstantIndexOp>(loc_, 0);
  }
  if (!idx) return {};

  // Extra byte-offset terms.
  for (const auto &t : acc.extraTerms) {
    Value v = scaled(valueOfReg(t.first), t.second);
    if (!v) return {};
    idx = builder_.create<arith::AddIOp>(idx, v).getResult();
  }

  // Constant byte offset.
  if (acc.constBytes != 0) {
    if (acc.constBytes % eb != 0) return {};
    long long c = acc.constBytes / eb;
    Value cv = builder_.create<arith::ConstantIndexOp>(loc_, c);
    idx = builder_.create<arith::AddIOp>(idx, cv).getResult();
  }

  return idx;
}

// The body of the kernel.
//
// PTX is a flat CFG; MLIR is structured. Everything below is one structuriser:
// a backward branch closes a loop, a forward conditional branch is a guard,
// and a range of instructions is emitted into the current block. The same
// routine runs at the top level and inside a loop body, which is why a guard
// INSIDE a loop (the shape of nvcc's tiled kernels) needed no second
// implementation.
LogicalResult PtxToRougeEmitter::emitBody() {
  const auto &body = fn_->body;
  // Pre-scan: find the backward branches that close a loop. A "bra $L" (or
  // "@%p bra $L") with L's position < i is a back edge; L is the loop header.
  // The PREDICATED back edge is the do-while form nvcc emits: the condition
  // sits at the bottom of the body, and the loop continues while it holds.
  loopBack_.clear();
  for (size_t i = 0; i < body.size(); ++i) {
    const PtxInstruction &ins = body[i];
    if (ins.op != "bra") continue;
    if (ins.args.empty()) continue;
    auto it = fn_->labels.find(ins.args[0]);
    if (it == fn_->labels.end()) continue;
    size_t target = (size_t)it->second;
    if (target < i) {
      BackEdge be;
      be.index = i;
      be.predicate = ins.pred;
      loopBack_[target] = be;
    }
  }
  return emitRange(0, body.size(), {});
}

// Is the register still read after 'from' before anything redefines it?
bool PtxToRougeEmitter::liveAfter(const std::string &reg, size_t from) const {
  const auto &body = fn_->body;
  for (size_t k = from; k < body.size(); ++k) {
    const PtxInstruction &ins = body[k];
    if (ins.args.empty()) continue;
    // The operands after the destination are reads. A store's first operand is
    // an address — a read too.
    for (size_t a = 1; a < ins.args.size(); ++a)
      if (regName(ins.args[a]) == reg) return true;
    if (!ins.args[0].empty() && ins.args[0][0] == '[' &&
        regName(ins.args[0]) == reg)
      return true;
    if (writesRegister(ins) && regName(ins.args[0]) == reg) return false;
  }
  return false;
}

// Emit [lo, hi): loops, guards and instructions, in the order they stand.
LogicalResult PtxToRougeEmitter::emitRange(size_t lo, size_t hi,
                                           const std::set<size_t> &skips) {
  const auto &body = fn_->body;
  size_t i = lo;
  while (i < hi) {
    // An instruction the enclosing loop owns: built from the loop's header and
    // its condition, not emitted where it stands.
    if (skips.count(i)) {
      ++i;
      continue;
    }

    // A loop: a backward branch closes the region [header, back_edge].
    auto lb = loopBack_.find(i);
    if (lb != loopBack_.end() && lb->second.index < hi) {
      if (failed(emitLoop(i, lb->second.index))) return failure();
      i = lb->second.index + 1;
      continue;
    }

    const PtxInstruction &ins = body[i];

    // A guard: "@%p bra L" — the threads that take the branch skip the block.
    // This is the shape of every range guard and of nvcc's if/else.
    if (ins.op == "bra" && !ins.pred.empty()) {
      if (ins.args.empty()) return fail("bra without a target");
      auto it = fn_->labels.find(ins.args[0]);
      if (it == fn_->labels.end())
        return fail("unknown label '" + ins.args[0] + "'");
      const size_t target = (size_t)it->second;
      if (target <= i || target > hi)
        return fail("the branch to '" + ins.args[0] + "' leaves the block: the "
                    "front-end structures only the branches that reconverge "
                    "inside it (a jump out of a loop is not supported)");

      Value pred = operandValue(ins.pred, builder_.getI1Type());
      if (!pred) return fail("the predicate of the branch is not understood");

      // Two shapes, and the difference is where the guarded block is:
      //
      //   @%p bra L;                  @%p bra L;
      //   <fallthrough block>         bra E;      <- the else path goes to E
      // L:                            L: <taken block>
      //                               E:
      //
      // In the first the FALLTHROUGH block runs when the predicate does not
      // hold; in the second the block after L is the taken path (nvcc's
      // if/else: the else branch jumps straight over it).
      size_t blockLo = i + 1;
      size_t blockHi = target;
      size_t resume = target;
      bool guardTaken = false;
      if (i + 1 < target && body[i + 1].op == "bra" &&
          body[i + 1].pred.empty()) {
        if (body[i + 1].args.empty())
          return fail("bra without a target");
        auto eit = fn_->labels.find(body[i + 1].args[0]);
        if (eit == fn_->labels.end())
          return fail("unknown label '" + body[i + 1].args[0] + "'");
        const size_t e = (size_t)eit->second;
        if (e < target)
          return fail("a backward branch inside a guarded block is not "
                      "supported");
        if (target > i + 2)
          return fail("unreachable code between an unconditional branch and "
                      "its target is not supported");
        blockLo = target;
        blockHi = std::min(e, hi);
        resume = blockHi;
        guardTaken = true;
      }

      // The values the block defines and the code after it still reads. The
      // scf.if carries them out; the block that does not run yields the value
      // from BEFORE the guard — which is what makes the pre-value (`mov %rs7,
      // %rs8` in front of `@%p4 bra L` in int8_gemm.ptx) matter: without it
      // PTX leaves the register undefined and the front-end refuses to guess.
      std::vector<std::string> carried;
      std::vector<std::string> definedInBlock;
      for (size_t k = blockLo; k < blockHi; ++k) {
        if (skip_.count(k)) continue;
        if (body[k].args.empty() || !writesRegister(body[k])) continue;
        const std::string r = regName(body[k].args[0]);
        if (r.empty()) continue;
        if (std::find(definedInBlock.begin(), definedInBlock.end(), r) ==
            definedInBlock.end())
          definedInBlock.push_back(r);
        if (std::find(carried.begin(), carried.end(), r) != carried.end())
          continue;
        // Live beyond the block. Scan to the end of the FUNCTION: inside a
        // loop body the value is also consumed by the loop's own yield, which
        // stands after the loop's range.
        if (liveAfter(r, blockHi)) carried.push_back(r);
      }
      SmallVector<Type> types;
      SmallVector<Value> outerVals;
      for (const std::string &r : carried) {
        Value v = valueOfReg(r);
        if (!v)
          return fail("the guarded block defines '" + r + "', the code after "
                      "it reads it, and there is no value before the block: "
                      "PTX leaves the register undefined there, the front-end "
                      "will not invent one");
        Type t = typeOfReg(r);
        if (!t) return fail("the type of '" + r + "' is not known");
        types.push_back(t);
        outerVals.push_back(v);
      }

      Value cond = pred;
      if (!guardTaken) {
        cond = builder_.create<arith::XOrIOp>(
            pred, builder_.create<arith::ConstantOp>(
                      builder_.getBoolAttr(true)))
                   .getResult();
      }
      auto ifOp = builder_.create<scf::IfOp>(TypeRange(types), cond,
                                             /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPointToStart(&ifOp.getThenRegion().front());
        if (failed(emitRange(blockLo, blockHi, skips))) return failure();
        SmallVector<Value> yields;
        for (const std::string &r : carried) {
          Value v = valueOfReg(r);
          if (!v)
            return fail("the guarded block does not define '" + r +
                        "', the code after it reads it");
          yields.push_back(v);
        }
        // Whether scf.if's builder left a terminator in the region depends on
        // the overload (it does for a result-less if, it does not for one with
        // result types), so ask the block itself. getTerminator() is not the
        // way: it returns the LAST OPERATION even when that is not a
        // terminator, and rewriting its operands silently corrupted the last
        // instruction of the block.
        Block &thenBlock = ifOp.getThenRegion().back();
        if (!thenBlock.empty() && isa<scf::YieldOp>(&thenBlock.back()))
          thenBlock.back().setOperands(yields);
        else {
          builder_.setInsertionPointToEnd(&thenBlock);
          builder_.create<scf::YieldOp>(yields);
        }
      }
      {
        Block &elseBlock = ifOp.getElseRegion().back();
        if (!elseBlock.empty() && isa<scf::YieldOp>(&elseBlock.back()))
          elseBlock.back().setOperands(outerVals);
        else {
          builder_.setInsertionPointToEnd(&elseBlock);
          builder_.create<scf::YieldOp>(outerVals);
        }
      }
      builder_.setInsertionPointAfter(ifOp);
      for (size_t c = 0; c < carried.size(); ++c)
        regValue_[carried[c]] = ifOp.getResult(c);
      // A register the block defines and does NOT carry out has no value
      // after the if: the path that skipped the block never wrote it. Keeping
      // the stale SSA value would make it look defined (and the verifier later
      // reports a dominance violation — block_reduce.ptx hit exactly that:
      // the value loaded inside the first guard was used as a loop's init).
      for (const std::string &r : definedInBlock)
        if (std::find(carried.begin(), carried.end(), r) == carried.end())
          regValue_.erase(r);
      i = resume;
      continue;
    }

    // An unconditional branch. Inside a range only "to the end of the range"
    // is understood: that is the last branch of an if/else block, and it says
    // nothing more than "the block ends here". Anything else jumps over code,
    // which the structuriser does not reorder.
    if (ins.op == "bra") {
      if (ins.args.empty()) return fail("bra without a target");
      auto it = fn_->labels.find(ins.args[0]);
      if (it == fn_->labels.end())
        return fail("unknown label '" + ins.args[0] + "'");
      if ((size_t)it->second == hi) return success();
      return fail("an unconditional branch to '" + ins.args[0] +
                  "' jumps over code: the front-end structures only branches "
                  "that reconverge (a loop is closed by a branch to its own "
                  "header)");
    }

    if (failed(emitInstruction(ins, i))) return failure();
    ++i;
  }
  return success();
}

// A loop: scf.while over the registers the body changes.
LogicalResult PtxToRougeEmitter::emitLoop(size_t header, size_t back) {
  const auto &body = fn_->body;
  const BackEdge be = loopBack_[header];

  // The condition of the loop. Two forms:
  //   * the back edge is PREDICATED (the do-while form nvcc emits): the loop
  //     continues while the predicate holds;
  //   * the back edge is unconditional: the loop has a forward exit branch,
  //     and the loop continues while its predicate does NOT hold.
  size_t exitBranch = body.size();
  std::string exitPred;
  std::string condPred = be.predicate;
  bool condInverted = false;
  if (condPred.empty()) {
    // Find the exit condition: a forward conditional branch inside
    // [header, back] whose target is beyond 'back'.
    for (size_t k = header; k < back; ++k) {
      const PtxInstruction &cand = body[k];
      if (cand.op == "bra" && !cand.pred.empty() && !cand.args.empty()) {
        auto lit = fn_->labels.find(cand.args[0]);
        if (lit != fn_->labels.end() && (size_t)lit->second > back) {
          exitBranch = k;
          exitPred = cand.pred;
          break;
        }
      }
    }
    if (exitPred.empty())
      return fail("a loop without an exit condition is not supported");
    condPred = exitPred;
    condInverted = true;
  }

  // The setp that defines the loop's predicate.
  size_t setpIdx = body.size();
  for (size_t k = header; k < back; ++k) {
    if (body[k].op.rfind("setp", 0) == 0 && !body[k].args.empty() &&
        regName(body[k].args[0]) == condPred) {
      setpIdx = k;
      break;
    }
  }
  if (setpIdx == body.size())
    return fail("the exit predicate of the loop is not a setp");

  // Loop-carried registers: defined before the loop and redefined inside it.
  std::set<std::string> definedBefore;
  for (size_t k = 0; k < header; ++k)
    if (!body[k].args.empty()) definedBefore.insert(regName(body[k].args[0]));
  std::set<std::string> definedInLoop;
  for (size_t k = header; k < back; ++k) {
    if (body[k].op == "bra") continue;
    if (!body[k].args.empty() && writesRegister(body[k]))
      definedInLoop.insert(regName(body[k].args[0]));
  }
  // A register redefined in the body is carried only if it has a value NOW.
  // A register without one is either recomputed in the body (the address
  // arithmetic, which is not emitted at all: it is part of the access) or it
  // is genuinely undefined in the first iteration — and then reading it in
  // the body fails loudly there, which is the honest report. Refusing a
  // register that the body simply overwrites first would reject
  // block_reduce.ptx: its loop re-does `cvta.to.shared` on every iteration.
  std::vector<std::string> carried;
  SmallVector<Value> initVals;
  for (const std::string &r : definedInLoop) {
    // The loop's own predicate is rebuilt by the condition every iteration:
    // it is not an input.
    if (r == condPred) continue;
    if (!definedBefore.count(r)) continue;
    Value v = valueOfReg(r);
    if (!v) continue;
    carried.push_back(r);
    initVals.push_back(v);
  }
  SmallVector<Type> carriedTypes;
  for (const std::string &r : carried) carriedTypes.push_back(typeOfReg(r));

  // A failure inside a region builder cannot be returned (the callback is
  // void), and a swallowed failure used to leave an EMPTY body block: the
  // module then failed verification with "'scf.while' op expects a non-empty
  // block", which says nothing about the real cause. The flag carries the
  // failure out.
  bool bodyFailed = false;
  auto beforeBuilder = [&](OpBuilder &b, Location loc, ValueRange args) {
    ImplicitLocOpBuilder saved = std::move(builder_);
    builder_ = ImplicitLocOpBuilder(loc_, b);
    for (size_t c = 0; c < carried.size(); ++c)
      regValue_[carried[c]] = args[c];
    if (failed(emitInstruction(body[setpIdx], setpIdx))) {
      bodyFailed = true;
      builder_ = std::move(saved);
      return;
    }
    Value p = valueOfReg(condPred);
    if (!p) {
      bodyFailed = true;
      builder_ = std::move(saved);
      return;
    }
    Value cond = p;
    if (condInverted) {
      // The exit branch: the loop continues while the predicate does NOT hold.
      cond = builder_.create<arith::XOrIOp>(
          p, builder_.create<arith::ConstantOp>(builder_.getBoolAttr(true)))
                 .getResult();
    }
    SmallVector<Value> condYield;
    for (const std::string &r : carried) condYield.push_back(valueOfReg(r));
    builder_.create<scf::ConditionOp>(cond, condYield);
    builder_ = std::move(saved);
  };

  auto afterBuilder = [&](OpBuilder &b, Location loc, ValueRange args) {
    ImplicitLocOpBuilder saved = std::move(builder_);
    builder_ = ImplicitLocOpBuilder(loc_, b);
    for (size_t c = 0; c < carried.size(); ++c)
      regValue_[carried[c]] = args[c];
    // The body is a RANGE like any other: guards inside it (nvcc's tiled
    // kernels guard the loads per thread) are structured here, not dropped.
    std::set<size_t> skips = {setpIdx, exitBranch};
    if (failed(emitRange(header, back, skips))) {
      bodyFailed = true;
      builder_ = std::move(saved);
      return;
    }
    SmallVector<Value> bodyYield;
    for (size_t c = 0; c < carried.size(); ++c) {
      Value v = valueOfReg(carried[c]);
      if (!v) v = args[c];
      bodyYield.push_back(v);
    }
    builder_.create<scf::YieldOp>(bodyYield);
    builder_ = std::move(saved);
  };

  auto whileOp = scf::WhileOp::create(builder_, carriedTypes, initVals,
                                      beforeBuilder, afterBuilder);
  if (bodyFailed)
    return fail("the loop body is not understood" +
                std::string(error_ ? ": " + *error_ : ""));

  // After the loop: update regValue_ with the results, and forget the
  // registers that only the body defined — they are not defined after the
  // loop (a trip count of zero never writes them).
  for (size_t c = 0; c < carried.size(); ++c)
    regValue_[carried[c]] = whileOp.getResult(c);
  for (const std::string &r : definedInLoop)
    if (std::find(carried.begin(), carried.end(), r) == carried.end())
      regValue_.erase(r);
  return success();
}

// One instruction.
LogicalResult PtxToRougeEmitter::emitInstruction(const PtxInstruction &ins,
                                                  size_t index) {
  const StringRef op = ins.op;

  // The address arithmetic is not emitted: the access is emitted with the
  // recovered structure instead.
  if (skip_.count(index)) return success();

  if (op == "bar.sync" || op == "barrier") {
    builder_.create<rouge::BarrierOp>();
    return success();
  }
  if (op == "ret" || op == "exit") return success();

  if (ins.args.empty()) return success();
  const std::string dst = regName(ins.args[0]);

  // A predicated instruction: the destination is written ONLY when the
  // predicate is true (PTX ISA 9.7.3). The value is computed either way and
  // selected.
  Value pred;
  if (!ins.pred.empty()) {
    pred = operandValue(ins.pred, builder_.getI1Type());
    if (!pred) return fail("the predicate of '" + op.str() + "' is not understood");
  }

  auto store = [&](Value v) {
    if (!pred) {
      regValue_[dst] = v;
      return;
    }
    // The old value, the new value and the predicate -> a select.
    Value old = valueOfReg(dst);
    if (!old) {
      Type t = v.getType();
      if (auto intTy = dyn_cast<IntegerType>(t))
        old = builder_.create<arith::ConstantOp>(
            builder_.getIntegerAttr(intTy, 0));
      else
        old = builder_.create<arith::ConstantOp>(
            builder_.getFloatAttr(t, 0.0));
    }
    Value sel = builder_.create<arith::SelectOp>(pred, v, old).getResult();
    regValue_[dst] = sel;
  };

  // A move.
  if (op.starts_with("mov.") || op == "mov") {
    if (ins.args.size() < 2) return fail("bad mov");
    Type t = typeOfReg(dst);
    Value v = operandValue(ins.args[1], t ? t : Type());
    if (!v) return fail("the source of the move '" + ins.args[1] + "' is not understood");
    if (t && v.getType() != t) {
      // The special registers (the thread id, the block id) are indices; a
      // move of them into a 32-bit register is a conversion.
      if (v.getType().isIndex() && isa<IntegerType>(t)) {
        v = builder_.create<arith::IndexCastOp>(t, v).getResult();
      } else {
        return fail("the move of '" + ins.args[1] + "' changes the type (" +
                    std::string(isa<IntegerType>(v.getType()) ? "int" : "other") +
                    " -> " +
                    (isa<IntegerType>(t) ? "int" : "other") +
                    "): use cvt");
      }
    }
    store(v);
    return success();
  }

  // A comparison: setp.
  if (op.starts_with("setp.") || op == "setp") {
    if (ins.args.size() < 3) return fail("bad setp");
    Type t = typeOfReg(regName(ins.args[1]));
    Value a = operandValue(ins.args[1], t);
    Value b = operandValue(ins.args[2], t);
    if (!a || !b) return fail("the operands of the comparison are not understood");
    Value cmp = emitCompare(op, a, b, t);
    if (!cmp) return fail("the comparison '" + op.str() + "' is not understood");
    regValue_[dst] = cmp;
    return success();
  }

  // A select: selp.type d, a, b, p -> p ? a : b. Both sources have the
  // destination's type (the predicate is the fourth operand).
  if (op.starts_with("selp.") || op == "selp") {
    if (ins.args.size() < 4) return fail("bad selp");
    Type t = typeOfReg(dst);
    Value a = operandValue(ins.args[1], t);
    Value b = operandValue(ins.args[2], t);
    Value p = operandValue(ins.args[3], IntegerType::get(module_.getContext(), 1));
    if (!a || !b || !p)
      return fail("the operands of the select are not understood ('" + op.str() +
                  "')");
    if (a.getType() != b.getType())
      return fail("the select mixes the types " +
                  std::string(a.getType() == t ? "of the destination" : "other") +
                  " ('" + op.str() + "')");
    Value v = builder_.create<arith::SelectOp>(p, a, b).getResult();
    store(v);
    return success();
  }

  // A conversion.
  if (op.starts_with("cvt.")) {
    if (ins.args.size() < 2) return fail("bad cvt");
    Type srcType = typeOfReg(regName(ins.args[1]));
    Value v = operandValue(ins.args[1], srcType);
    if (!v) return fail("the source of the conversion is not understood");
    Type dstType = typeOfReg(dst);
    if (!dstType) return fail("the type of the conversion's result is not known");
    Value res;
    if (isa<IntegerType>(v.getType()) && isa<IntegerType>(dstType)) {
      unsigned sw = cast<IntegerType>(v.getType()).getWidth();
      unsigned dw = cast<IntegerType>(dstType).getWidth();
      // The signedness of the destination decides the extension.
      bool destSigned = op.find(".u") == StringRef::npos;
      if (sw < dw) {
        if (destSigned)
          res = builder_.create<arith::ExtSIOp>(dstType, v).getResult();
        else
          res = builder_.create<arith::ExtUIOp>(dstType, v).getResult();
      } else if (sw > dw) {
        res = builder_.create<arith::TruncIOp>(dstType, v).getResult();
      } else {
        res = v;
      }
    } else if (isa<FloatType>(v.getType()) && isa<FloatType>(dstType)) {
      // The float-to-float conversions: cvt.f32.f16 (the f16 dot product
      // widens its accumulator, gemm_tile.ptx) and cvt.f16.f32.
      const unsigned sw = cast<FloatType>(v.getType()).getWidth();
      const unsigned dw = cast<FloatType>(dstType).getWidth();
      if (sw < dw)
        res = builder_.create<arith::ExtFOp>(dstType, v).getResult();
      else if (sw > dw)
        res = builder_.create<arith::TruncFOp>(dstType, v).getResult();
      else
        res = v;
    } else if (isa<IntegerType>(v.getType()) && isa<FloatType>(dstType)) {
      const bool isUnsigned = op.find(".u") != StringRef::npos;
      res = isUnsigned
                ? builder_.create<arith::UIToFPOp>(dstType, v).getResult()
                : builder_.create<arith::SIToFPOp>(dstType, v).getResult();
    } else if (isa<FloatType>(v.getType()) && isa<IntegerType>(dstType)) {
      const bool isUnsigned = op.find(".u") != StringRef::npos;
      res = isUnsigned
                ? builder_.create<arith::FPToUIOp>(dstType, v).getResult()
                : builder_.create<arith::FPToSIOp>(dstType, v).getResult();
    } else {
      return fail("the conversion '" + op.str() + "' is not supported");
    }
    regValue_[dst] = res;
    return success();
  }

  // A parameter read: ld.param.u64 %rd, [name].
  if (op.starts_with("ld.param.")) {
    if (ins.args.size() < 2) return fail("bad ld.param");
    std::string name = ins.args[1];
    if (name.size() > 1 && name.front() == '[' && name.back() == ']')
      name = name.substr(1, name.size() - 2);
    int pidx = -1;
    for (size_t i = 0; i < fn_->params.size(); ++i)
      if (fn_->params[i].name == name) pidx = (int)i;
    if (pidx < 0) return fail("unknown parameter '" + name + "'");
    // A pointer parameter: the register holds the address, which the
    // analysis already used; the access is emitted with the recovered
    // structure, so the read itself is not emitted.
    if (paramIsPointer_.count(std::to_string(pidx))) {
      skip_.insert(index);
      return success();
    }
    Value v = paramValue_[std::to_string(pidx)];
    if (!v) return fail("the parameter '" + name + "' is not known");
    Type want = typeOfReg(dst);
    if (want && v.getType() != want) {
      if (auto ity = dyn_cast<IntegerType>(v.getType()))
        if (auto wty = dyn_cast<IntegerType>(want))
          v = builder_.create<arith::TruncIOp>(wty, v).getResult();
    }
    regValue_[dst] = v;
    return success();
  }

  // A memory access.
  auto it = access_.find(index);
  if (it != access_.end() && it->second.valid) {
    const MemAccess &acc = it->second;
    Value base;
    if (acc.sharedIndex >= 0) {
      if (acc.sharedIndex >= (int)sharedValue_.size())
        return fail("the shared variable is out of range");
      base = sharedValue_[(size_t)acc.sharedIndex];
    } else {
      base = paramValue_[std::to_string(acc.paramIndex)];
    }
    if (!base) return fail("the base of the access is not known");
    Value idx = accessIndex(acc);
    if (!idx) {
      // Name the shape of the address: "not understood" alone does not say
      // WHICH term is wrong.
      std::string shape = "index '" + acc.indexReg + "' scale " +
                          std::to_string(acc.scale);
      for (const auto &t : acc.extraTerms)
        shape += ", + '" + t.first + "' scale " + std::to_string(t.second);
      return fail("the index of the access is not understood (" + shape +
                  ", element size " + std::to_string(acc.elemBytes) +
                  "): the front-end recovers only param + index*element size");
    }
    Type elem = acc.elemType;
    if (op.starts_with("ld.")) {
      Value v = builder_.create<memref::LoadOp>(elem, base, ValueRange{idx});
      regValue_[dst] = v;
      return success();
    }
    // Atomics: atom.* / red.* -> memref.atomic_rmw.
    if (op.starts_with("atom.") || op.starts_with("red.")) {
      const bool hasDst = op.starts_with("atom.");
      if (op.find(".cas.") != std::string::npos)
        return fail("atom.cas is not supported in the MLIR frontend yet");
      const size_t valArg = hasDst ? 2 : 1;
      if (ins.args.size() < valArg + 1) return fail("bad atomic");
      arith::AtomicRMWKind kind;
      const bool isMin = op.find(".min.") != std::string::npos;
      const bool isMax = op.find(".max.") != std::string::npos;
      const bool isExch = op.find(".exch.") != std::string::npos;
      const bool isFloat = elem.isF32() || elem.isF64();
      const bool isUnsigned = op.find(".u32") != std::string::npos ||
                              op.find(".u64") != std::string::npos;
      if (isExch) kind = arith::AtomicRMWKind::assign;
      else if (op.find(".add.") != std::string::npos)
        kind = isFloat ? arith::AtomicRMWKind::addf : arith::AtomicRMWKind::addi;
      else if (isMin)
        kind = isFloat ? arith::AtomicRMWKind::minimumf
                       : (isUnsigned ? arith::AtomicRMWKind::minu
                                     : arith::AtomicRMWKind::mins);
      else if (isMax)
        kind = isFloat ? arith::AtomicRMWKind::maximumf
                       : (isUnsigned ? arith::AtomicRMWKind::maxu
                                     : arith::AtomicRMWKind::maxs);
      else return fail("unsupported atomic kind '" + op.str() + "'");
      Value val = operandValue(ins.args[valArg], elem);
      if (!val) return fail("the value of the atomic is not understood");
      auto rmw = builder_.create<memref::AtomicRMWOp>(
          elem, kind, val, base, ValueRange{idx});
      if (hasDst) regValue_[dst] = rmw.getResult();
      return success();
    }
    // A store.
    if (ins.args.size() < 2) return fail("bad store");
    Value v = operandValue(ins.args[1], elem);
    if (!v) return fail("the value of the store is not understood");
    builder_.create<memref::StoreOp>(v, base, ValueRange{idx});
    return success();
  }

  // The arithmetic.
  {
    const std::string &o = op.str();
    auto bin = [&](auto opTag) -> Value {
      using OpTy = decltype(opTag);
      if (ins.args.size() < 3) return {};
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      if (!a || !b) return {};
      return builder_.create<OpTy>(a, b).getResult();
    };

    // The float and the integer forms are different arith ops: an
    // arith.addi on f32 does not verify, and silently treating add.f32 as an
    // integer addition is exactly the sort of mistranslation the front-end
    // must not make. vadd.ptx (the first real nvcc-shaped kernel through the
    // front-end) exercises them.
    if (o.rfind("add.", 0) == 0 || o.rfind("sub.", 0) == 0) {
      const bool isAdd = o.rfind("add.", 0) == 0;
      Type t = typeOfReg(dst);
      Value v;
      if (isa<FloatType>(t)) {
        v = isAdd ? bin(arith::AddFOp{}) : bin(arith::SubFOp{});
      } else {
        v = isAdd ? bin(arith::AddIOp{}) : bin(arith::SubIOp{});
      }
      if (!v) return fail("the addition/subtraction is not understood ('" +
                          op.str() + "')");
      store(v);
      return success();
    }
    if (o.rfind("mul.hi.", 0) == 0)
      return fail("mul.hi is not supported by the MLIR front-end yet");
    if (o.rfind("mul.", 0) == 0) {
      if (ins.args.size() < 3) return fail("bad mul");
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      if (!a || !b) return fail("the multiplication is not understood ('" + op.str() + "')");
      Value v;
      if (isa<FloatType>(t)) {
        v = builder_.create<arith::MulFOp>(a, b).getResult();
      } else if (o.rfind("mul.wide.", 0) == 0) {
        // The wide multiplication: 32x32 -> 64.
        Type i64 = IntegerType::get(module_.getContext(), 64);
        Value a64 = builder_.create<arith::ExtUIOp>(i64, a).getResult();
        Value b64 = builder_.create<arith::ExtUIOp>(i64, b).getResult();
        v = builder_.create<arith::MulIOp>(a64, b64).getResult();
      } else {
        v = builder_.create<arith::MulIOp>(a, b).getResult();
      }
      store(v);
      return success();
    }
    if (o.rfind("div.", 0) == 0) {
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      if (!a || !b) return fail("the division is not understood ('" + op.str() + "')");
      Value v;
      if (isa<FloatType>(t))
        v = builder_.create<arith::DivFOp>(a, b).getResult();
      else if (o.find(".u") != std::string::npos)
        v = builder_.create<arith::DivUIOp>(a, b).getResult();
      else
        v = builder_.create<arith::DivSIOp>(a, b).getResult();
      store(v);
      return success();
    }
    if (o.rfind("rem.", 0) == 0) {
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      if (!a || !b) return fail("the remainder is not understood ('" + op.str() + "')");
      Value v;
      if (isa<FloatType>(t))
        v = builder_.create<arith::RemFOp>(a, b).getResult();
      else if (o.find(".u") != std::string::npos)
        v = builder_.create<arith::RemUIOp>(a, b).getResult();
      else
        v = builder_.create<arith::RemSIOp>(a, b).getResult();
      store(v);
      return success();
    }
    if (o.rfind("min.", 0) == 0 || o.rfind("max.", 0) == 0) {
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      if (!a || !b) return fail("the min/max is not understood ('" + op.str() + "')");
      if (isa<FloatType>(t))
        return fail("min/max of a float is not supported by the MLIR "
                    "front-end yet");
      bool isMax = o.rfind("max.", 0) == 0;
      bool isUnsigned = o.find(".u") != std::string::npos;
      Value v;
      if (isUnsigned)
        v = isMax ? builder_.create<arith::MaxUIOp>(a, b).getResult()
                  : builder_.create<arith::MinUIOp>(a, b).getResult();
      else
        v = isMax ? builder_.create<arith::MaxSIOp>(a, b).getResult()
                  : builder_.create<arith::MinSIOp>(a, b).getResult();
      store(v);
      return success();
    }
    if (o.rfind("and.", 0) == 0 || o.rfind("or.", 0) == 0 ||
        o.rfind("xor.", 0) == 0) {
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      if (!a || !b) return fail("the bitwise operation is not understood");
      Value v;
      if (o.rfind("and.", 0) == 0)
        v = builder_.create<arith::AndIOp>(a, b).getResult();
      else if (o.rfind("or.", 0) == 0)
        v = builder_.create<arith::OrIOp>(a, b).getResult();
      else
        v = builder_.create<arith::XOrIOp>(a, b).getResult();
      store(v);
      return success();
    }
    if (o.rfind("not.", 0) == 0) {
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      if (!a) return fail("the bitwise not is not understood ('" + op.str() + "')");
      auto intTy = dyn_cast<IntegerType>(t);
      if (!intTy) return fail("the bitwise not of a float is not supported");
      Value ones = builder_.create<arith::ConstantOp>(
          loc_, builder_.getIntegerAttr(intTy, -1));
      Value v = builder_.create<arith::XOrIOp>(a, ones).getResult();
      store(v);
      return success();
    }
    if (o.rfind("shl.", 0) == 0 || o.rfind("shr.", 0) == 0) {
      Type t = typeOfReg(dst);
      auto intTy = dyn_cast<IntegerType>(t);
      if (!intTy) return fail("a shift of a non-integer is not supported");
      const unsigned bits = intTy.getWidth();
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      if (!a || !b) return fail("the shift is not understood ('" + op.str() + "')");
      // PTX ISA 9.7.9.8/9.7.9.9: "shift amounts greater than the register
      // width N are clamped to N". arith.shli/shrui/shrai are POISON from N
      // on (they lower to LLVM's shifts), so the clamp is explicit here:
      // shift by (b & (N-1)), which is always legal, and select the
      // saturated result when b >= N (0 for shl/shr.u, the sign fill for
      // shr.s). The amount is a .u32 value regardless of the width.
      if (b.getType() != t) {
        auto bTy = dyn_cast<IntegerType>(b.getType());
        if (!bTy) return fail("the shift amount is not an integer");
        b = bTy.getWidth() < bits
                ? builder_.create<arith::ExtUIOp>(loc_, t, b).getResult()
                : builder_.create<arith::TruncIOp>(loc_, t, b).getResult();
      }
      auto iconst = [&](uint64_t v) -> Value {
        return builder_.create<arith::ConstantOp>(
            loc_, builder_.getIntegerAttr(intTy, v));
      };
      const bool isShl = o.rfind("shl.", 0) == 0;
      const bool isSigned = !isShl && o.find(".s") != std::string::npos;
      Value sh = builder_.create<arith::AndIOp>(loc_, b, iconst(bits - 1));
      Value raw = isShl ? builder_
                              .create<arith::ShLIOp>(loc_, a, sh)
                              .getResult()
                  : isSigned
                      ? builder_.create<arith::ShRSIOp>(loc_, a, sh).getResult()
                      : builder_.create<arith::ShRUIOp>(loc_, a, sh).getResult();
      Value big = builder_.create<arith::CmpIOp>(
          loc_, arith::CmpIPredicate::uge, b, iconst(bits));
      Value sat = isSigned
                      ? builder_.create<arith::ShRSIOp>(loc_, a, iconst(bits - 1))
                            .getResult()
                      : iconst(0);
      Value v = builder_.create<arith::SelectOp>(loc_, big, sat, raw);
      store(v);
      return success();
    }
    if (o.rfind("popc.", 0) == 0) {
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      if (!a) return fail("the population count is not understood");
      Value v = builder_.create<math::CtPopOp>(a).getResult();
      store(v);
      return success();
    }
    if (o.rfind("clz.", 0) == 0) {
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      if (!a) return fail("the count of leading zeros is not understood");
      // The leading zeros of the 32-bit value: the width is the type's.
      auto intTy = dyn_cast<IntegerType>(t);
      if (!intTy) return fail("the count of leading zeros needs an integer");
      Value v = builder_.create<math::CountLeadingZerosOp>(a).getResult();
      store(v);
      return success();
    }
    if (o.rfind("dp4a.", 0) == 0) {
      // dp4a.atype.btype d, a, b, c: the four bytes of a and of b (signed
      // or unsigned per the type), the products summed into 64 bits, plus
      // the accumulator c.
      if (ins.args.size() < 4) return fail("bad dp4a");
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      Value acc = operandValue(ins.args[3], t);
      if (!a || !b || !acc) return fail("the dp4a is not understood");
      auto intTy = dyn_cast<IntegerType>(t);
      if (!intTy) return fail("the dp4a needs an integer");
      // The signedness of the two operands: "dp4a.s32.s32" and
      // "dp4a.u32.u32" (the type suffix of the op).
      const bool aSigned = o.find(".s32") != std::string::npos;
      const bool bSigned = o.find(".s32.s32") != std::string::npos;
      Type i64 = IntegerType::get(module_.getContext(), 64);
      Value sum = builder_.create<arith::ExtSIOp>(i64, acc).getResult();
      Value mask = builder_.create<arith::ConstantOp>(
          loc_, builder_.getIntegerAttr(intTy, 0xff));
      Value signMask = builder_.create<arith::ConstantOp>(
          loc_, builder_.getIntegerAttr(intTy, 0x80));
      for (int byte = 0; byte < 4; ++byte) {
        unsigned shift = byte * 8;
        // The byte of the operand, sign-extended to 32 bits: the classic
        // "(x ^ 0x80) - 0x80" (the sign bit is flipped, and the subtraction
        // turns the flipped sign into a borrow).
        auto extract = [&](Value src, bool isSigned) -> Value {
          Value raw = builder_.create<arith::ShRUIOp>(src, constInt(t, shift))
                          .getResult();
          raw = builder_.create<arith::AndIOp>(raw, mask).getResult();
          if (!isSigned) return raw;
          Value flipped = builder_.create<arith::XOrIOp>(raw, signMask).getResult();
          return builder_.create<arith::SubIOp>(flipped, signMask).getResult();
        };
        Value av = extract(a, aSigned);
        Value bv = extract(b, bSigned);
        Value product = builder_.create<arith::MulIOp>(
            builder_.create<arith::ExtSIOp>(i64, av).getResult(),
            builder_.create<arith::ExtSIOp>(i64, bv).getResult())
            .getResult();
        sum = builder_.create<arith::AddIOp>(sum, product).getResult();
      }
      // The sum is 64 bits; the destination may be narrower (then the low
      // part is taken, as the 32-bit register holds).
      if (intTy.getWidth() == 32)
        sum = builder_.create<arith::TruncIOp>(intTy, sum).getResult();
      regValue_[dst] = sum;
      return success();
    }
    // ---- bfi / bfe: bit-field insert / extract (PTX ISA 9.7.9) ----------
    //
    //   bfi.type f, a, b, c, d;   pos = c & 0xff, len = d & 0xff
    //       f = b; for (i < len && pos+i <= msb) f[pos+i] = a[i];
    //   bfe.type d, a, b, c;      pos = b & 0xff, len = c & 0xff
    //       the field a[pos .. pos+len-1], zero- or sign-extended; len == 0
    //       gives 0.
    //
    // The position and the length have to be COMPILE-TIME constants here:
    // the clamps ("pos+i <= msb") and the sign bit would otherwise need a
    // chain of selects whose only purpose is to be correct for a dynamic
    // value, and a wrong clamp is a silent wrong answer. nvcc emits
    // immediates in practice; a dynamic one is refused by name.
    auto constantOf = [](Value v) -> std::optional<int64_t> {
      if (auto c = v.getDefiningOp<arith::ConstantOp>())
        if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
          return i.getInt();
      return std::nullopt;
    };
    if (o.rfind("bfi.", 0) == 0 || o.rfind("bfe.", 0) == 0) {
      const bool insert = o.rfind("bfi.", 0) == 0;
      const size_t need = insert ? 5u : 4u;
      if (ins.args.size() < need)
        return fail(insert ? "bad bfi (expected f, a, b, pos, len)"
                           : "bad bfe (expected d, a, pos, len)");
      Type t = typeOfReg(dst);
      auto intTy = dyn_cast<IntegerType>(t);
      if (!intTy) return fail("bfi/bfe need an integer register");
      const unsigned bits = intTy.getWidth();
      Value a = operandValue(ins.args[1], t);
      if (!a) return fail("the bfi/bfe source is not understood");
      const size_t posArg = insert ? 3 : 2;
      Value posV = operandValue(ins.args[posArg], t);
      Value lenV = operandValue(ins.args[posArg + 1], t);
      auto posOpt = constantOf(posV);
      auto lenOpt = constantOf(lenV);
      if (!posOpt || !lenOpt)
        return fail("a dynamic bfi/bfe position or length is not supported: "
                    "the clamps to the word have to be known at compile time");
      const unsigned pos = (unsigned)(*posOpt & 0xff);
      const unsigned len = (unsigned)(*lenOpt & 0xff);

      if (insert) {
        Value b = operandValue(ins.args[2], t);
        if (!b) return fail("the bfi base is not understood");
        // Nothing is written past the msb, and a position past it inserts
        // nothing at all: f stays b.
        const unsigned effLen = (pos >= bits) ? 0u : std::min(len, bits - pos);
        if (effLen == 0) {
          store(b);
          return success();
        }
        const uint64_t low =
            (effLen >= bits) ? ~uint64_t{0} : ((uint64_t{1} << effLen) - 1);
        Value mask = constInt(t, (int64_t)low);
        Value shMask = builder_.create<arith::ShLIOp>(mask, constInt(t, pos))
                           .getResult();
        Value keep =
            builder_.create<arith::AndIOp>(
                b, builder_.create<arith::XOrIOp>(
                       shMask, constInt(t, -1)).getResult())
                .getResult();
        Value field =
            builder_.create<arith::AndIOp>(
                builder_.create<arith::ShLIOp>(a, constInt(t, pos)).getResult(),
                shMask)
                .getResult();
        store(builder_.create<arith::OrIOp>(keep, field).getResult());
        return success();
      }

      // bfe: the field is empty when len == 0, and the ISA's answer for
      // anything past the msb is the sign replicated over the whole word.
      const bool signedField = o.find(".s") != StringRef::npos;
      if (len == 0) {
        store(constInt(t, 0));
        return success();
      }
      if (pos >= bits) {
        if (!signedField) {
          store(constInt(t, 0));
          return success();
        }
        Value sbit = builder_.create<arith::AndIOp>(
            builder_.create<arith::ShRUIOp>(a, constInt(t, (int64_t)bits - 1))
                .getResult(),
            constInt(t, 1))
                         .getResult();
        store(builder_.create<arith::SubIOp>(constInt(t, 0), sbit).getResult());
        return success();
      }
      const unsigned effLen = std::min(len, bits - pos);
      const uint64_t low =
          (effLen >= bits) ? ~uint64_t{0} : ((uint64_t{1} << effLen) - 1);
      Value field = builder_.create<arith::AndIOp>(
          builder_.create<arith::ShRUIOp>(a, constInt(t, pos)).getResult(),
          constInt(t, (int64_t)low))
                        .getResult();
      if (!signedField || effLen >= bits) {
        store(field);
        return success();
      }
      // Sign-extend: fill the bits above the field with its top bit.
      Value sbit = builder_.create<arith::AndIOp>(
          builder_.create<arith::ShRUIOp>(a, constInt(t, pos + effLen - 1))
              .getResult(),
          constInt(t, 1))
                       .getResult();
      Value fill = builder_.create<arith::ShLIOp>(
          builder_.create<arith::SubIOp>(constInt(t, 0), sbit).getResult(),
          constInt(t, effLen))
                       .getResult();
      store(builder_.create<arith::OrIOp>(field, fill).getResult());
      return success();
    }

    // ---- prmt.b32: byte permute, mode 0 (PTX ISA 9.7.10.7) --------------
    //
    // The eight source bytes are the 64-bit value {b, a} from the low end:
    // 0..3 are a's, 4..7 are b's. Each 4-bit field of the selector picks the
    // byte for d.b0..d.b3; a field with bit 3 set replicates the selected
    // byte's SIGN over the whole byte. Bits 19:16 of the selector are the
    // MODE, and only mode 0 is implemented here.
    //
    // The selector must be a constant: the mode decides the meaning of the
    // whole instruction, and a dynamic selector cannot be checked. Guessing
    // would be a silent wrong answer for every mode-1 selector.
    if (o.rfind("prmt.", 0) == 0) {
      if (ins.args.size() < 4) return fail("bad prmt");
      Type t = typeOfReg(dst);
      auto intTy = dyn_cast<IntegerType>(t);
      if (!intTy || intTy.getWidth() != 32)
        return fail("prmt is supported only on 32-bit words");
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      Value selV = operandValue(ins.args[3], t);
      if (!a || !b || !selV) return fail("the prmt is not understood");
      auto selOpt = constantOf(selV);
      if (!selOpt)
        return fail("a dynamic prmt selector is not supported: the mode field "
                    "(bits 19:16) cannot be checked");
      const unsigned sel = (unsigned)*selOpt;
      const unsigned mode = (sel >> 16) & 0xfu;
      if (mode != 0)
        return fail("the prmt mode " + std::to_string(mode) +
                    " is not supported (only mode 0)");
      Type i64 = IntegerType::get(module_.getContext(), 64);
      Value wa = builder_.create<arith::ExtUIOp>(i64, a).getResult();
      Value wb = builder_.create<arith::ExtUIOp>(i64, b).getResult();
      Value wide = builder_.create<arith::OrIOp>(
          wa, builder_.create<arith::ShLIOp>(wb, constInt(i64, 32)).getResult())
                       .getResult();
      Value res = constInt(t, 0);
      for (unsigned i = 0; i < 4; ++i) {
        const unsigned field = (sel >> (4 * i)) & 0xfu;
        const unsigned idx = field & 7u;
        // The byte itself: truncation alone would keep a's HIGH bytes too,
        // so they would leak into the neighbouring output bytes after the
        // shift back up. Mask to 8 bits.
        Value byte =
            builder_.create<arith::AndIOp>(
                builder_.create<arith::TruncIOp>(
                    t, builder_.create<arith::ShRUIOp>(
                           wide, constInt(i64, idx * 8)).getResult())
                    .getResult(),
                constInt(t, 0xff))
                .getResult();
        if (field & 8u) {
          // The sign of the selected byte, replicated: 0xff or 0x00.
          Value sbit = builder_.create<arith::AndIOp>(
              builder_.create<arith::ShRUIOp>(byte, constInt(t, 7)).getResult(),
              constInt(t, 1))
                           .getResult();
          byte = builder_.create<arith::AndIOp>(
              builder_.create<arith::SubIOp>(constInt(t, 0), sbit).getResult(),
              constInt(t, 0xff))
                     .getResult();
        }
        res = builder_.create<arith::OrIOp>(
            res, builder_.create<arith::ShLIOp>(byte, constInt(t, 8 * i))
                     .getResult())
                  .getResult();
      }
      store(res);
      return success();
    }

    if (o.rfind("neg.", 0) == 0) {
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      if (!a) return fail("the negation is not understood ('" + op.str() + "')");
      auto intTy = dyn_cast<IntegerType>(t);
      if (!intTy) return fail("the negation of a float is not supported");
      Value zero = builder_.create<arith::ConstantOp>(
          loc_, builder_.getIntegerAttr(intTy, 0));
      Value v = builder_.create<arith::SubIOp>(zero, a).getResult();
      store(v);
      return success();
    }
    if (o.rfind("mad.", 0) == 0) {
      if (ins.args.size() < 4) return fail("bad mad");
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      Value c = operandValue(ins.args[3], t);
      if (!a || !b || !c) return fail("the mad is not understood ('" + op.str() + "')");
      Value mul = builder_.create<arith::MulIOp>(a, b).getResult();
      Value v = builder_.create<arith::AddIOp>(mul, c).getResult();
      store(v);
      return success();
    }
  }

  return fail("the instruction '" + op.str() + "' is not supported by the "
              "PTX front-end of the rouge dialect");
}

// The comparison of a setp.
Value PtxToRougeEmitter::emitCompare(StringRef op, Value a, Value b, Type type) {
  StringRef pred = op;
  // Strip the "setp." prefix.
  if (pred.starts_with("setp.")) pred = pred.substr(5);
  bool isUnsigned = pred.find(".u") != StringRef::npos;
  bool isFloat = isa<FloatType>(type);
  // The predicate: the first word.
  StringRef p = pred.split('.').first;
  arith::CmpIPredicate ip;
  if (p == "eq") ip = arith::CmpIPredicate::eq;
  else if (p == "ne") ip = arith::CmpIPredicate::ne;
  else if (p == "lt") ip = isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
  else if (p == "le") ip = isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle;
  else if (p == "gt") ip = isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
  else if (p == "ge") ip = isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge;
  else return {};
  if (isFloat) {
    arith::CmpFPredicate fp;
    if (p == "eq") fp = arith::CmpFPredicate::OEQ;
    else if (p == "ne") fp = arith::CmpFPredicate::ONE;
    else if (p == "lt") fp = arith::CmpFPredicate::OLT;
    else if (p == "le") fp = arith::CmpFPredicate::OLE;
    else if (p == "gt") fp = arith::CmpFPredicate::OGT;
    else if (p == "ge") fp = arith::CmpFPredicate::OGE;
    else return {};
    return builder_.create<arith::CmpFOp>(fp, a, b).getResult();
  }
  return builder_.create<arith::CmpIOp>(ip, a, b).getResult();
}

// Deletes the ops of the function that are pure and whose results are not
// used. The front-end emits EVERY PTX instruction, including the address
// arithmetic (which is already folded into the accesses) and the registers
// that the kernel never reads; without this the loop body carries dead
// index casts that the vectoriser has to reason about. Only ops with no
// side effects are touched: a load stays even if its result is unused (the
// front-end does not have to prove that the read was pointless).
void eraseDeadPureOps(Operation *root) {
  bool changed = true;
  while (changed) {
    changed = false;
    root->walk([&](Operation *op) {
      if (op == root || !op->use_empty() || op->getNumRegions() != 0) return;
      // A terminator (func.return is `Pure` as well) is what keeps the block
      // valid: an unused return is not dead code.
      if (op->hasTrait<OpTrait::IsTerminator>()) return;
      if (!isMemoryEffectFree(op)) return;
      op->erase();
      changed = true;
    });
  }
}

// One kernel.
LogicalResult PtxToRougeEmitter::emitKernel(const PtxFunction &fn) {
  fn_ = &fn;
  regValue_.clear();
  regType_.clear();
  ptrValue_.clear();
  paramValue_.clear();
  sharedValue_.clear();
  access_.clear();
  skip_.clear();

  // The analysis first: it decides the types of the parameters, so it has
  // to run before the signature is built.
  if (failed(analyseAddresses())) return failure();
  if (failed(inferTypes())) return failure();
  if (failed(buildSignature())) return failure();

  // The shared memory: an allocation inside the kernel, in the shared
  // address space. The lowering turns it into a private per-block storage.
  if (!fn.sharedVars.empty()) {
    for (size_t si = 0; si < fn.sharedVars.size(); ++si) {
      const PtxSharedVar &sv = fn.sharedVars[si];
      // A dynamic (.extern, unsized) declaration has no size until launch,
      // and the launch would have to pass it: refused, not guessed.
      if (sv.dynamic)
        return fail("the dynamic .shared variable '" + sv.name +
                    "' is not supported: its size is a launch parameter");
      Attribute space = builder_.getIntegerAttr(
          IntegerType::get(module_.getContext(), 32), 3);
      // The element type of a shared variable is not in the PTX; the
      // accesses decide. Use the type of the accesses to it.
      Type elem = Float32Type::get(module_.getContext());
      for (const auto &kv : access_) {
        if (kv.second.sharedIndex == (int)si) {
          elem = kv.second.elemType;
          break;
        }
      }
      // The declaration counts in ITS element size (".b8 smem[256]" is 256
      // BYTES), while the memref counts in the element type of the accesses.
      // Getting this wrong UNDER-allocates whenever the declaration is
      // coarser than the access (".b64 s[8]" read as i32).
      const int64_t declBytes = (int64_t)sv.count * (int64_t)sv.elemBytes;
      const int64_t elemBytes = bytesOf(elem);
      if (elemBytes <= 0 || declBytes <= 0 || declBytes % elemBytes != 0)
        return fail("the size of the shared variable '" + sv.name +
                    "' is not a multiple of its element size");
      MemRefType mt = MemRefType::get({declBytes / elemBytes}, elem,
                                      MemRefLayoutAttrInterface(), space);
      auto alloc = builder_.create<memref::AllocOp>(mt);
      sharedValue_.push_back(alloc.getResult());
    }
  }

  if (failed(emitBody())) return failure();
  builder_.create<func::ReturnOp>();
  eraseDeadPureOps(func_);

  // The host function with the launch: the launch configuration belongs to
  // the caller, so the host takes the kernel's arguments and launches the
  // kernel over the grid.
  if (hints_.block > 0) {
    ImplicitLocOpBuilder hostBuilder(loc_, module_.getContext());
    hostBuilder.setInsertionPointAfter(func_);
    SmallVector<Type> hostInputs(func_.getFunctionType().getInputs());
    auto host = func::FuncOp::create(
        hostBuilder, fn_->name + "_host",
        FunctionType::get(module_.getContext(), hostInputs, /*results=*/{}));
    Block *hostEntry = host.addEntryBlock();
    ImplicitLocOpBuilder hb(loc_, module_.getContext());
    hb.setInsertionPointToStart(hostEntry);

    Value grid = hb.create<arith::ConstantIndexOp>(hints_.grid).getResult();
    Value one = hb.create<arith::ConstantIndexOp>(1).getResult();
    Value block = hb.create<arith::ConstantIndexOp>(hints_.block).getResult();
    SmallVector<Value> operands;
    for (BlockArgument a : hostEntry->getArguments()) operands.push_back(a);
    hb.create<rouge::LaunchOp>(
        SymbolRefAttr::get(module_.getContext(), fn_->name), grid, one, one,
        block, block, block, operands);
    hb.create<func::ReturnOp>();
  }
  return success();
}

}  // namespace

LogicalResult translatePtxToRouge(ModuleOp module, const PtxProgram &program,
                                  const PtxShapeHints &hints, std::string *error) {
  // The dialects whose operations the front-end creates must be LOADED:
  // creating an operation of an unregistered dialect is a crash, not an
  // error (in a release build the check is an assertion).
  MLIRContext *ctx = module.getContext();
  ctx->getOrLoadDialect<rouge::RougeDialect>();
  ctx->getOrLoadDialect<arith::ArithDialect>();
  ctx->getOrLoadDialect<func::FuncDialect>();
  ctx->getOrLoadDialect<math::MathDialect>();
  ctx->getOrLoadDialect<memref::MemRefDialect>();
  ctx->getOrLoadDialect<scf::SCFDialect>();

  PtxToRougeEmitter emitter(module, program, hints, error);
  return emitter.run();
}

}  // namespace rouge
