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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace mlir;

namespace rouge {
namespace {

// ---------------------------------------------------------------------------
// The address model
// ---------------------------------------------------------------------------

// What the analysis knows about the value in a register.
struct PtrValue {
  enum Kind { None, ParamPtr, Address, Offset } kind = None;
  // ParamPtr: the value of a kernel parameter (a pointer).
  int paramIndex = -1;
  // Address / Offset: base + index * scale.
  int addrParam = -1;      // the kernel parameter the address is based on
  int addrShared = -1;     // the shared variable the address is based on
  std::string indexReg;    // the register with the element index ("" = none)
  int scale = 1;           // the byte scale of the index
  bool isGlobal = true;    // global or shared
};

// A memory access, recovered from the address arithmetic.
struct MemAccess {
  bool valid = false;
  int paramIndex = -1;
  int sharedIndex = -1;
  std::string indexReg;
  int scale = 1;
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

  LogicalResult emitKernel(const PtxFunction &fn);
  LogicalResult inferTypes();
  LogicalResult analyseAddresses();
  LogicalResult buildSignature();
  LogicalResult emitBody();
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
LogicalResult PtxToRougeEmitter::analyseAddresses() {
  MLIRContext *ctx = module_.getContext();

  // The parameters: their index by name.
  llvm::StringMap<int> paramIndex;
  for (size_t i = 0; i < fn_->params.size(); ++i)
    paramIndex[fn_->params[i].name] = (int)i;

  // The shared variables: their index by name.
  llvm::StringMap<int> sharedIndex;
  for (size_t i = 0; i < fn_->sharedVars.size(); ++i)
    sharedIndex[fn_->sharedVars[i].name] = (int)i;

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
      bool isImm = !scaleTok.empty() && (std::isdigit((unsigned char)scaleTok[0]) ||
                                        scaleTok[0] == '-');
      if (isImm) {
        scale = strtol(scaleTok.c_str(), nullptr, 10);
      } else {
        // The scale is not a constant: this is not an address offset but an
        // ordinary 64-bit value (a global index, for example). It is emitted
        // as it is.
        continue;
      }
      PtrValue pv;
      pv.kind = PtrValue::Offset;
      pv.indexReg = idx;
      pv.scale = (int)scale;
      ptrValue_[regName(ins.args[0])] = pv;
      skip_.insert(i);
      continue;
    }

    // add.s64 %rd, %a, %b -> the base plus the offset.
    if (op == "add.s64" || op == "add.u64") {
      if (ins.args.size() < 3) return fail("bad add.s64");
      const std::string a = regName(ins.args[1]);
      const std::string b = regName(ins.args[2]);
      auto ai = ptrValue_.find(a);
      auto bi = ptrValue_.find(b);
      PtrValue pv;
      if (ai != ptrValue_.end() && ai->second.kind == PtrValue::Address &&
          bi != ptrValue_.end() && bi->second.kind == PtrValue::Offset) {
        pv = ai->second;
        pv.indexReg = bi->second.indexReg;
        pv.scale = bi->second.scale;
      } else if (bi != ptrValue_.end() && bi->second.kind == PtrValue::Address &&
                 ai != ptrValue_.end() && ai->second.kind == PtrValue::Offset) {
        pv = bi->second;
        pv.indexReg = ai->second.indexReg;
        pv.scale = ai->second.scale;
      } else {
        // A plain 64-bit addition: an ordinary value.
        continue;
      }
      ptrValue_[regName(ins.args[0])] = pv;
      skip_.insert(i);
      continue;
    }

    // A conversion that keeps an offset: the index becomes the new
    // register (the value may be truncated, which the PTX allows).
    if (op.starts_with("cvt.")) {
      if (ins.args.size() < 2) return fail("bad cvt");
      const std::string src = regName(ins.args[1]);
      auto it = ptrValue_.find(src);
      if (it != ptrValue_.end() && it->second.kind == PtrValue::Offset) {
        PtrValue pv = it->second;
        pv.indexReg = regName(ins.args[0]);
        ptrValue_[regName(ins.args[0])] = pv;
        skip_.insert(i);
        continue;
      }
      continue;
    }

    // A memory access: the address register must be understood.
    if (op.starts_with("ld.") && op.find(".param.") == StringRef::npos) {
      if (ins.args.size() < 2) return fail("bad load");
      const std::string addr = regName(stripBrackets(ins.args[1]));
      auto it = ptrValue_.find(addr);
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
      acc.isGlobal = it->second.isGlobal;
      acc.elemType = typeFromSuffix(typeSuffix(op), ctx);
      if (!acc.elemType)
        return fail("the element type of the load '" + op.str() + "' is not known");
      acc.elemBytes = bytesOf(acc.elemType);
      access_[i] = acc;
      continue;
    }
    if (op.starts_with("st.") && op.find(".param.") == StringRef::npos) {
      if (ins.args.size() < 2) return fail("bad store");
      const std::string addr = regName(stripBrackets(ins.args[0]));
      auto it = ptrValue_.find(addr);
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
      acc.isGlobal = it->second.isGlobal;
      acc.elemType = typeFromSuffix(typeSuffix(op), ctx);
      if (!acc.elemType)
        return fail("the element type of the store '" + op.str() + "' is not known");
      acc.elemBytes = bytesOf(acc.elemType);
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
  // A number?
  bool numeric = true;
  for (char c : tok)
    if (!std::isdigit((unsigned char)c) && c != '-' && c != '+') numeric = false;
  if (numeric && !tok.empty() && tok[0] != '-') {
    APInt value(64, 0);
    if (!tok.getAsInteger(10, value)) {
      if (auto intTy = dyn_cast<IntegerType>(type))
        return builder_.create<arith::ConstantOp>(
            builder_.getIntegerAttr(intTy, value.trunc(intTy.getWidth())));
      if (isa<Float32Type>(type) || isa<Float64Type>(type))
        return builder_.create<arith::ConstantOp>(
            builder_.getFloatAttr(type, value.getSExtValue()));
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
  // A special register (%tid.x, %ctaid.x, %ntid.x, %nctaid.x).
  std::string name = regName(tok);
  if (name.size() > 2 && name.compare(name.size() - 2, 2, ".x") == 0)
    name = name.substr(0, name.size() - 2);
  if (name == "tid") {
    auto op = builder_.create<rouge::ThreadIdOp>(
        loc_, builder_.getIndexType(),
        rouge::DimensionAttr::get(module_.getContext(), rouge::Dimension::x));
    return op.getResult();
  }
  if (name == "ctaid") {
    auto op = builder_.create<rouge::BlockIdOp>(
        loc_, builder_.getIndexType(),
        rouge::DimensionAttr::get(module_.getContext(), rouge::Dimension::x));
    return op.getResult();
  }
  if (name == "ntid") {
    auto op = builder_.create<rouge::BlockDimOp>(
        loc_, builder_.getIndexType(),
        rouge::DimensionAttr::get(module_.getContext(), rouge::Dimension::x));
    return op.getResult();
  }
  if (name == "nctaid") {
    auto op = builder_.create<rouge::GridDimOp>(
        loc_, builder_.getIndexType(),
        rouge::DimensionAttr::get(module_.getContext(), rouge::Dimension::x));
    return op.getResult();
  }
  // An ordinary register.
  return valueOfReg(name);
}

// The index of an access, in elements.
Value PtxToRougeEmitter::accessIndex(const MemAccess &acc) {
  Value raw;
  if (!acc.indexReg.empty()) raw = valueOfReg(acc.indexReg);
  if (!raw) {
    // No index: the first element.
    auto intTy = dyn_cast<IntegerType>(acc.elemType);
    if (intTy)
      raw = builder_.create<arith::ConstantOp>(
          loc_, builder_.getIntegerAttr(intTy, 0));
  }
  if (!raw) return {};
  // The scale must be exactly the element size: the front-end does not
  // divide (a division would break the vectoriser, which needs the index to
  // be "the thread plus a constant").
  if (acc.scale != acc.elemBytes) return {};
  // The index: an integer -> an index.
  if (raw.getType().isIndex()) return raw;
  if (auto intTy = dyn_cast<IntegerType>(raw.getType())) {
    if (intTy.getWidth() == 1) return {};
    return builder_.create<arith::IndexCastOp>(builder_.getIndexType(), raw);
  }
  return {};
}

// The body of the kernel: the instructions, with the early exits turned into
// scf.if.
LogicalResult PtxToRougeEmitter::emitBody() {
  const auto &body = fn_->body;
  size_t i = 0;
  while (i < body.size()) {
    const PtxInstruction &ins = body[i];

    // An early exit: "@%p bra $L" — the threads that take the branch skip
    // the block up to the label. This is the shape of every range guard.
    if (ins.op == "bra" && !ins.pred.empty()) {
      if (ins.args.empty()) return fail("bra without a target");
      auto it = fn_->labels.find(ins.args[0]);
      if (it == fn_->labels.end())
        return fail("unknown label '" + ins.args[0] + "'");
      size_t target = (size_t)it->second;
      if (target <= i)
        return fail("only a forward branch is supported (the loop forms are "
                    "not translated yet)");
      if (target > body.size()) return fail("the label is out of range");

      // The predicate.
      Value pred = operandValue(ins.pred, builder_.getI1Type());
      if (!pred) return fail("the predicate of the branch is not understood");
      Value notPred =
          builder_.create<arith::XOrIOp>(
              pred,
              builder_.create<arith::ConstantOp>(builder_.getBoolAttr(true)))
              .getResult();

      // The guarded block: the instructions up to the label.
      SmallVector<Operation *> saved;
      (void)saved;
      // The values the block defines must not be used after it: the scf.if
      // would have to carry them, and that is not implemented yet.
      std::set<std::string> defined;
      for (size_t k = i + 1; k < target; ++k)
        if (!body[k].args.empty()) defined.insert(regName(body[k].args[0]));
      for (size_t k = target; k < body.size(); ++k) {
        for (const std::string &a : body[k].args) {
          std::string name = regName(a);
          if (defined.count(name))
            return fail("a value defined inside a guarded block is used "
                        "after it: the front-end does not carry values out "
                        "of an early exit yet");
        }
      }

      // The else region: the threads that take the branch skip the block.
      // The builder adds the terminators of both regions, so nothing is
      // added here.
      auto ifOp = builder_.create<scf::IfOp>(notPred, /*withElseRegion=*/true);
      OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPointToStart(&ifOp.getThenRegion().front());
      for (size_t k = i + 1; k < target; ++k) {
        if (failed(emitInstruction(body[k], k))) return failure();
      }
      builder_.setInsertionPointAfter(ifOp);
      i = target;
      continue;
    }

    if (failed(emitInstruction(ins, i))) return failure();
    ++i;
  }
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
    } else if (isa<Float32Type>(v.getType()) && isa<Float32Type>(dstType)) {
      res = v;
    } else if (isa<IntegerType>(v.getType()) && isa<Float32Type>(dstType)) {
      res = builder_.create<arith::SIToFPOp>(dstType, v).getResult();
    } else if (isa<Float32Type>(v.getType()) && isa<IntegerType>(dstType)) {
      res = builder_.create<arith::FPToSIOp>(dstType, v).getResult();
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
    if (!idx)
      return fail("the index of the access is not understood: the front-end "
                  "recovers only param + index*element size (the scale is "
                  "not the element size)");
    Type elem = acc.elemType;
    if (op.starts_with("ld.")) {
      Value v = builder_.create<memref::LoadOp>(elem, base, ValueRange{idx});
      regValue_[dst] = v;
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

    if (o.rfind("add.", 0) == 0) {
      Value v = bin(arith::AddIOp{});
      if (!v) return fail("the addition is not understood ('" + op.str() + "')");
      store(v);
      return success();
    }
    if (o.rfind("sub.", 0) == 0) {
      Value v = bin(arith::SubIOp{});
      if (!v) return fail("the subtraction is not understood ('" + op.str() + "')");
      store(v);
      return success();
    }
    if (o.rfind("mul.lo.", 0) == 0 || o.rfind("mul.wide.", 0) == 0) {
      if (ins.args.size() < 3) return fail("bad mul");
      Type t = typeOfReg(dst);
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      if (!a || !b) return fail("the multiplication is not understood ('" + op.str() + "')");
      Value v;
      if (o.rfind("mul.wide.", 0) == 0) {
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
      if (o.find(".u") != std::string::npos)
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
      if (o.find(".u") != std::string::npos)
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
      Value a = operandValue(ins.args[1], t);
      Value b = operandValue(ins.args[2], t);
      if (!a || !b) return fail("the shift is not understood ('" + op.str() + "')");
      Value v;
      if (o.rfind("shl.", 0) == 0)
        v = builder_.create<arith::ShLIOp>(a, b).getResult();
      else if (o.find(".u") != std::string::npos)
        v = builder_.create<arith::ShRUIOp>(a, b).getResult();
      else
        v = builder_.create<arith::ShRSIOp>(a, b).getResult();
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
      Attribute space = builder_.getIntegerAttr(
          IntegerType::get(module_.getContext(), 32), 3);
      MemRefType t = MemRefType::get(
          {sv.count}, Float32Type::get(module_.getContext()),
          MemRefLayoutAttrInterface(), space);
      // The element type of a shared variable is not in the PTX; the
      // accesses decide. Use the type of the accesses to it.
      Type elem = Float32Type::get(module_.getContext());
      for (const auto &kv : access_) {
        if (kv.second.sharedIndex == (int)si) {
          elem = kv.second.elemType;
          break;
        }
      }
      MemRefType mt =
          MemRefType::get({sv.count}, elem, MemRefLayoutAttrInterface(), space);
      auto alloc = builder_.create<memref::AllocOp>(mt);
      sharedValue_.push_back(alloc.getResult());
    }
  }

  if (failed(emitBody())) return failure();
  builder_.create<func::ReturnOp>();

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
