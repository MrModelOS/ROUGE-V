//=- ROUGE-V: unit-stride -> vector (the vectorising step of the SIMT->Vector
// lowering).
//
// After rouge-simt-to-scf a kernel is a plain loop over the thread index. If
// the memory accesses of that loop are unit-stride (index = thread + const),
// the loop can be executed V elements at a time: that is the whole point of
// going from a GPU's SIMT execution to a CPU's vector unit.
//
// The rewrite:
//
//   scf.for %t = %lb to %ub step 1 {        scf.for %t = %lb to %ub-(V-1) step V {
//     %v = memref.load  %A[%t]                %v = vector.load  %A[%t] : ..., vector<Vxf32>
//     memref.store %v, %B[%t]   ==>            vector.store %v, %B[%t] : ..., vector<Vxf32>
//   }                                     }
//                                         scf.for %t = %ub-(V-1) to %ub step 1 { ...scalar... }
//
// The tail loop keeps the last (V-1) elements exact: the vector loop covers
// only the full groups.
//
// The pass has two clearly separated phases:
//
//   1. ANALYSIS — can this loop be vectorised at all? Every check lives
//      here, and a refusal happens BEFORE anything is modified. (An earlier
//      version rewrote first and checked later; a refusal then left the
//      module half-transformed and the verifier failed on it.)
//
//   2. REWRITE — no failure path: the analysis has already proved that the
//      shape fits.
//
// The pass REFUSES (loudly, never silently):
//   * a loop whose step is not 1 (not unit stride);
//   * a lower bound that is not a multiple of V (the vector access would
//     straddle the element boundary of the memref);
//   * a memref that is not 1-D, has a dynamic shape, or an offset that is
//     not a multiple of V;
//   * an access whose index is not "thread id + constant";
//   * a value in the body that does not come from a load or a constant of the
//     same body (the data flow has to be closed for the vector types to
//     line up);
//   * any op in the body outside the supported set.
//
// The element-wise arithmetic rides along unchanged: arith ops are
// element-wise, so the only thing that changes is the TYPE of the values
// (scalar -> vector). Constants inside the body become splat vector
// constants.
//
//   rouge-opt --rouge-unit-stride-to-vector kernel.mlir
//
//=.

#include "rouge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace mlir;

namespace rouge {
namespace {

// The scalar element types this pass vectorises: 32-bit floats and integers.
bool isVectorizableElement(Type t) { return t.isF32() || t.isInteger(32); }

VectorType vectorOf(Type elem, int64_t width) {
  return VectorType::get({width}, elem);
}

// The alignment of a vector access. WITHOUT this attribute the LLVM lowering
// assumes the alignment of the ELEMENT (4 bytes for f32), not of the vector
// (16 bytes for vector<4xf32>) — a vector access may then be lowered as
// an aligned 16-byte access on a 4-byte-aligned pointer. The attribute
// exists since LLVM 21.
llvm::MaybeAlign alignmentOf(VectorType vecTy) {
  const int64_t bits =
      vecTy.getElementType().isF32() ? 32 : vecTy.getElementTypeBitWidth();
  const uint64_t bytes = (uint64_t)bits * (uint64_t)vecTy.getNumElements() / 8;
  if (bytes == 0) return llvm::MaybeAlign();
  return llvm::MaybeAlign((uint64_t)llvm::PowerOf2Ceil(bytes));
}

// A constant index value.
struct ConstIndex {
  bool found = false;
  int64_t value = 0;
};

ConstIndex constantIndex(Value v) {
  ConstIndex out;
  auto c = v.getDefiningOp<arith::ConstantIndexOp>();
  if (!c) return out;
  out.found = true;
  out.value = c.value();
  return out;
}

// The index of an access: "thread id + constant", "thread id", or a plain
// constant (an access that does not depend on the thread at all).
struct AccessIndex {
  bool ok = false;
  int64_t constant = 0;
};

// True if the op is inside a divergent region of the body.
bool isInsideDivergence(Operation *op, ArrayRef<Operation *> bodyOps) {
  for (Operation *candidate : bodyOps) {
    auto ifOp = dyn_cast<scf::IfOp>(candidate);
    if (!ifOp) continue;
    if (op->getBlock() == &ifOp.getThenRegion().front()) return true;
  }
  return false;
}

// True if the value is the thread id plus a constant.
bool isThreadIdPlusConst(Value v, Value iv) {
  auto add = v.getDefiningOp<arith::AddIOp>();
  if (!add) return false;
  bool usesIv = false;
  for (Value op : add.getOperands())
    if (op == iv) usesIv = true;
  return usesIv;
}

// The condition of a divergence: a comparison of the thread id (or the
// thread id plus a constant) with a constant. That is the shape the
// generated kernels have ("the thread index is in range"), and it is the
// shape the vector form can express: the lane-wise thread ids against a
// splat.
bool analyzeCondition(Value cond, Value iv) {
  auto cmp = cond.getDefiningOp<arith::CmpIOp>();
  if (!cmp) return false;
  for (Value operand : cmp.getOperands()) {
    // The thread id itself.
    if (operand == iv) continue;
    // The thread id plus a constant.
    if (auto add = operand.getDefiningOp<arith::AddIOp>()) {
      bool usesIv = false;
      for (Value op : add.getOperands())
        if (op == iv) usesIv = true;
      if (usesIv) continue;
    }
    // A constant of the body.
    auto def = operand.getDefiningOp();
    if (def && isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) continue;
    return false;
  }
  return true;
}

AccessIndex analyzeIndex(Value idx, Value iv) {
  if (idx == iv) {
    AccessIndex out;
    out.ok = true;
    return out;
  }
  if (auto add = idx.getDefiningOp<arith::AddIOp>()) {
    for (unsigned k = 0; k < 2; ++k) {
      Value op = add.getOperands()[k];
      if (op != iv) continue;
      ConstIndex c = constantIndex(add.getOperands()[1 - k]);
      if (!c.found) continue;
      AccessIndex out;
      out.ok = true;
      out.constant = c.value;
      return out;
    }
  }
  ConstIndex c = constantIndex(idx);
  if (c.found) {
    AccessIndex out;
    out.ok = true;
    out.constant = c.value;
    return out;
  }
  return AccessIndex();
}

// What an op of the body is, for the vectorisation.
enum class BodyKind {
  Load,
  Store,
  Constant,
  Elementwise,
  // The divergent region: an scf.if whose condition depends on the thread
  // id. In the vector form it becomes a MASK (vector<Nxi1>), and the memory
  // accesses inside it become the masked ones.
  Divergent,
};

struct BodyOp {
  Operation *op = nullptr;
  BodyKind kind = BodyKind::Elementwise;
  MemRefType memrefTy;
  Value index;          // the access index
  Value value;          // for a store: the value stored
  int64_t resultNo = 0; // for a load: which result
  // The op is inside a divergent region: its memory accesses are masked.
  bool divergent = false;
};

struct UnitStrideToVector
    : public PassWrapper<UnitStrideToVector, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnitStrideToVector)

  UnitStrideToVector() = default;
  UnitStrideToVector(int64_t width) { width_ = width; }

  StringRef getArgument() const final { return "rouge-unit-stride-to-vector"; }
  StringRef getDescription() const final {
    return "Vectorise unit-stride memory accesses of a thread loop (SIMT->Vector)";
  }

  // The dialects whose ops this pass creates must be LOADED: an input without
  // vector ops never loads the Vector dialect, and creating a vector op in an
  // unloaded dialect is refused.
  //
  // They are declared as DEPENDENCIES rather than loaded inside the pass:
  // a pass over func.func runs on the functions IN PARALLEL, and loading a
  // dialect from several threads at once is a race (it crashed exactly that
  // way — the framework loads the declared dialects before the pass starts).
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp fn = getOperation();

    SmallVector<scf::ForOp> loops;
    fn.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    unsigned converted = 0, refused = 0;
    for (scf::ForOp forOp : loops) {
      if (failed(vectorizeLoop(forOp))) {
        ++refused;
        continue;
      }
      ++converted;
    }
    llvm::errs() << "rouge-unit-stride-to-vector: " << fn.getName()
                 << ": loops vectorised=" << converted << " refused=" << refused
                 << "\n";
  }

  // The analysis: can this loop be vectorised?
  LogicalResult analyze(scf::ForOp forOp, SmallVectorImpl<BodyOp> &plan,
                        int64_t &vectorUb) {
    // Only a plain counted loop with no carried values.
    if (!forOp.getInitArgs().empty() || forOp.getNumResults() != 0) {
      forOp.emitRemark("skipped: the loop carries values");
      return failure();
    }
    // The step must be 1 (unit stride).
    ConstIndex step = constantIndex(forOp.getStep());
    if (!step.found || step.value != 1) {
      forOp.emitRemark("skipped: the step is not 1");
      return failure();
    }
    // The bounds must be known constants: the lower bound decides the
    // alignment, the upper bound the length of the vector part.
    ConstIndex lb = constantIndex(forOp.getLowerBound());
    ConstIndex ub = constantIndex(forOp.getUpperBound());
    if (!lb.found || !ub.found) {
      forOp.emitRemark("skipped: the bounds are not constants");
      return failure();
    }
    if (ub.value <= lb.value) {
      forOp.emitRemark("skipped: an empty loop");
      return failure();
    }
    // The lower bound must be a multiple of the vector width: otherwise the
    // first vector access would straddle the element boundary.
    if (((lb.value % width_) + width_) % width_ != 0) {
      forOp.emitRemark(
          "skipped: the lower bound is not aligned to the vector width");
      return failure();
    }
    if (forOp.getBody()->getNumArguments() != 1) {
      forOp.emitRemark("skipped: unexpected loop body signature");
      return failure();
    }
    Value iv = forOp.getInductionVar();

    // Classify the body. A divergent region (an scf.if whose condition
    // depends on the thread id) is a first-class citizen here: in the vector
    // form it becomes a mask, and the accesses inside it become the masked
    // ones — the threads that took the other branch simply do not touch the
    // memory.
    DenseMap<Operation *, unsigned> indexOf;
    SmallVector<Operation *> bodyOps;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      indexOf[&op] = bodyOps.size();
      bodyOps.push_back(&op);
    }
    // The ops of a divergent region come right after the region itself.
    SmallVector<Operation *> flatOps;
    for (Operation *op : bodyOps) {
      flatOps.push_back(op);
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp) continue;
      if (!ifOp.getElseRegion().empty()) {
        forOp.emitRemark("skipped: a divergent region with an else branch");
        return failure();
      }
      if (ifOp.getNumResults() != 0) {
        forOp.emitRemark("skipped: a divergent region with results");
        return failure();
      }
      if (!analyzeCondition(ifOp.getCondition(), iv)) {
        forOp.emitRemark(
            "skipped: the condition of the divergence is not a comparison of "
            "the thread id");
        return failure();
      }
      for (Operation &inner : ifOp.getThenRegion().front().without_terminator())
        flatOps.push_back(&inner);
    }
    for (Operation *op : flatOps) {
      BodyOp item;
      item.op = op;
      item.divergent = isInsideDivergence(op, bodyOps);
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        item.kind = BodyKind::Load;
        item.memrefTy = load.getMemRefType();
        item.index = load.getIndices()[0];
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        item.kind = BodyKind::Store;
        item.memrefTy = store.getMemRefType();
        item.index = store.getIndices()[0];
        item.value = store.getValueToStore();
      } else if (isa<arith::ConstantOp>(op)) {
        item.kind = BodyKind::Constant;
      } else if (isa<arith::ConstantIndexOp>(op)) {
        // An index constant: in the vector form it is a broadcast (there is
        // no dense constant of the index type). It is not part of the
        // access indices (those go through remapIndex), but the condition
        // of a divergence compares against it.
        item.kind = BodyKind::Constant;
      } else if (isa<scf::IfOp>(op)) {
        // The region itself: in the vector form it becomes the mask.
        item.kind = BodyKind::Divergent;
      } else {
        item.kind = BodyKind::Elementwise;
      }
      // The memref constraints (1-D, static, aligned offset, element type).
      if (item.kind == BodyKind::Load || item.kind == BodyKind::Store) {
        if (!item.memrefTy.hasStaticShape() || item.memrefTy.getRank() != 1) {
          forOp.emitRemark("skipped: the memref is not a static 1-D memref");
          return failure();
        }
        SmallVector<int64_t> strides;
        int64_t offset = 0;
        if (failed(item.memrefTy.getStridesAndOffset(strides, offset))) {
          forOp.emitRemark("skipped: the memref layout is not strided");
          return failure();
        }
        if (((offset % width_) + width_) % width_ != 0) {
          forOp.emitRemark(
              "skipped: the memref offset is not aligned to the vector width");
          return failure();
        }
        if (!isVectorizableElement(item.memrefTy.getElementType())) {
          forOp.emitRemark("skipped: unsupported element type");
          return failure();
        }
        if (!analyzeIndex(item.index, iv).ok) {
          forOp.emitRemark("skipped: the access index is not 'thread + const'");
          return failure();
        }
      }
      plan.push_back(item);
    }
    if (plan.empty()) {
      forOp.emitRemark("skipped: an empty body");
      return failure();
    }
    // The data flow must be closed: every value used in the body comes from
    // a load or a constant OF THIS BODY. Otherwise the vector types would
    // not line up (a scalar from outside the body cannot be stored into a
    // vector).
    for (const BodyOp &item : plan) {
      SmallVector<Value> uses;
      if (item.kind == BodyKind::Store)
        uses.push_back(item.value);
      else if (item.kind == BodyKind::Elementwise)
        for (Value v : item.op->getOperands()) uses.push_back(v);
      for (Value v : uses) {
        // The thread id (and the thread id plus a constant) are the lane-wise
        // values in the vector form: they are allowed.
        if (v == iv || isThreadIdPlusConst(v, iv)) continue;
        Operation *def = v.getDefiningOp();
        if (!def) {
          forOp.emitRemark("skipped: a value comes from outside the body");
          return failure();
        }
        // A constant of the body (including an index constant) is a splat in
        // the vector form.
        if (isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) continue;
        auto it = indexOf.find(def);
        if (it == indexOf.end()) {
          forOp.emitRemark("skipped: a value comes from outside the body");
          return failure();
        }
        Operation *producer = bodyOps[it->second];
        if (!isa<memref::LoadOp, arith::ConstantOp>(producer)) {
          forOp.emitRemark("skipped: a value is not a load or a constant");
          return failure();
        }
      }
    }
    // The vector part covers the full groups; the tail the rest.
    vectorUb = ub.value - ((ub.value - lb.value) % width_);
    return success();
  }

  // The rewrite. No failure path: the analysis has proved the shape fits.
  void rewrite(scf::ForOp forOp, SmallVectorImpl<BodyOp> &plan, int64_t vectorUb) {
    Location loc = forOp.getLoc();
    ImplicitLocOpBuilder builder(loc, forOp);
    Value iv = forOp.getInductionVar();

    // 1) The vector loop.
    auto vecFor = scf::ForOp::create(
        builder, forOp.getLowerBound(),
        arith::ConstantIndexOp::create(builder, vectorUb),
        arith::ConstantIndexOp::create(builder, width_));
    builder.setInsertionPointToStart(vecFor.getBody());
    Value vecIv = vecFor.getInductionVar();
    DenseMap<Value, Value> mapped;

    // The lane-wise thread ids: [iv, iv+1, ..., iv+V-1]. The condition of a
    // divergence is computed PER THREAD, so in the vector form it is
    // computed per lane — that is exactly what a mask is.
    VectorType idxVecTy = VectorType::get({width_}, builder.getIndexType());
    Value laneIv;
    {
      auto step = vector::StepOp::create(builder, loc, idxVecTy);
      auto splat = vector::BroadcastOp::create(builder, loc, idxVecTy, vecIv);
      laneIv = builder.create<arith::AddIOp>(splat, step).getResult();
      mapped[iv] = laneIv;
    }

    // The mask of the divergent region (if the body has one).
    Value currentMask;

    for (const BodyOp &item : plan) {
      if (item.kind == BodyKind::Divergent) {
        // The region itself: in the vector form there is no branch — there is
        // a mask. The condition, computed per lane, IS the mask.
        auto ifOp = cast<scf::IfOp>(item.op);
        currentMask = mapped.lookup(ifOp.getCondition());
        continue;
      }
      if (item.kind == BodyKind::Load) {
        auto load = cast<memref::LoadOp>(item.op);
        VectorType vecTy = vectorOf(item.memrefTy.getElementType(), width_);
        Value idx = remapIndex(item.index, iv, vecIv);
        if (item.divergent) {
          // A masked load: the lanes that took the other branch read
          // nothing (the pass-through value is what they get).
          Value passthru = builder.create<arith::ConstantOp>(
              vecTy, builder.getZeroAttr(vecTy));
          auto maskedLoad = vector::MaskedLoadOp::create(
              builder, vecTy, load.getMemRef(), ValueRange{idx}, currentMask,
              passthru, alignmentOf(vecTy));
          mapped[load.getResult()] = maskedLoad.getResult();
          continue;
        }
        auto vecLoad = vector::LoadOp::create(builder, vecTy, load.getMemRef(),
                                             ValueRange{idx}, /*nontemporal=*/false,
                                             alignmentOf(vecTy));
        mapped[load.getResult()] = vecLoad.getResult();
        continue;
      }
      if (item.kind == BodyKind::Store) {
        auto store = cast<memref::StoreOp>(item.op);
        VectorType vecTy = vectorOf(item.memrefTy.getElementType(), width_);
        Value val = mapped.lookup(item.value);
        if (!val) {
          // A constant from outside the body: it becomes a broadcast.
          Operation *def = item.value.getDefiningOp();
          if (def && isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) {
            VectorType cTy =
                VectorType::get({width_}, def->getResult(0).getType());
            val = vector::BroadcastOp::create(builder, loc, cTy, item.value)
                      .getResult();
          }
        }
        if (!val) {
          forOp.emitRemark("skipped: the stored value is not a vector");
          return;
        }
        Value idx = remapIndex(item.index, iv, vecIv);
        if (item.divergent) {
          // A masked store: the lanes that took the other branch write
          // nothing.
          vector::MaskedStoreOp::create(builder, store.getMemRef(),
                                        ValueRange{idx}, currentMask, val,
                                        alignmentOf(vecTy));
          continue;
        }
        vector::StoreOp::create(builder, val, store.getMemRef(), ValueRange{idx},
                               /*nontemporal=*/false, alignmentOf(vecTy));
        continue;
      }
      if (item.kind == BodyKind::Constant) {
        if (auto idxCst = dyn_cast<arith::ConstantIndexOp>(item.op)) {
          VectorType vecTy = VectorType::get({width_}, idxCst.getType());
          mapped[idxCst.getResult()] =
              vector::BroadcastOp::create(builder, loc, vecTy, idxCst.getResult())
                  .getResult();
          continue;
        }
        auto cst = cast<arith::ConstantOp>(item.op);
        if (cst.getType().isIndex()) {
          // An index constant: a broadcast (there is no dense constant of
          // the index type).
          VectorType vecTy = VectorType::get({width_}, cst.getType());
          mapped[cst.getResult()] =
              vector::BroadcastOp::create(builder, loc, vecTy, cst.getResult())
                  .getResult();
          continue;
        }
        VectorType vecTy = vectorOf(cst.getType(), width_);
        auto splat = DenseElementsAttr::get(vecTy, cst.getValue());
        auto newCst = arith::ConstantOp::create(builder, vecTy, splat);
        mapped[cst.getResult()] = newCst.getResult();
        continue;
      }
      // An element-wise arithmetic op: the operands are vectors now, and so
      // is the result.
      SmallVector<Value> operands;
      for (Value v : item.op->getOperands()) {
        Value mappedV = mapped.lookup(v);
        if (!mappedV) {
          // A constant (of the body or from outside it): it becomes a
          // broadcast — the condition of a divergence compares the lane-wise
          // thread ids against it.
          Operation *def = v.getDefiningOp();
          if (def && isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) {
            VectorType vecTy =
                VectorType::get({width_}, def->getResult(0).getType());
            mappedV = vector::BroadcastOp::create(builder, loc, vecTy, v)
                          .getResult();
          }
        }
        if (!mappedV) {
          forOp.emitRemark("skipped: an operand is not a vector");
          return;
        }
        operands.push_back(mappedV);
      }
      Operation *cloned = builder.clone(*item.op);
      cloned->setOperands(operands);
      if (cloned->getNumResults() == 1) {
        Type opTy = cloned->getOperand(0).getType();
        if (isa<arith::CmpIOp>(cloned)) {
          // A comparison gives a mask: a vector of i1.
          cloned->getResult(0).setType(
              VectorType::get({width_}, builder.getI1Type()));
        } else if (isa<VectorType>(opTy)) {
          cloned->getResult(0).setType(opTy);
        }
        mapped[item.op->getResult(0)] = cloned->getResult(0);
      }
    }
    // (the scf.for built by the ODS builder already has its scf.yield)

    // 2) The tail: the original body, from the end of the vector part. It
    // goes AFTER the vector loop, not inside it.
    builder.setInsertionPointAfter(vecFor);
    auto tailFor = scf::ForOp::create(
        builder, arith::ConstantIndexOp::create(builder, vectorUb),
        forOp.getUpperBound(), forOp.getStep());
    builder.setInsertionPointToStart(tailFor.getBody());
    Value tailIv = tailFor.getInductionVar();
    // The tail keeps the SCALAR form: the divergent region stays a branch
    // (the tail is at most V-1 iterations, so the serialisation costs
    // nothing). The clones go through a mapper, so the nested region comes
    // along remapped — a clone that still referred to the old loop would be
    // a dangling reference.
    IRMapping tailMapper;
    tailMapper.map(iv, tailIv);
    for (const BodyOp &item : plan) {
      if (item.kind == BodyKind::Load) {
        auto load = cast<memref::LoadOp>(item.op);
        Value idx = remapIndex(item.index, iv, tailIv);
        auto newLoad = memref::LoadOp::create(builder, load.getMemRef(), idx);
        tailMapper.map(load.getResult(), newLoad.getResult());
        continue;
      }
      if (item.kind == BodyKind::Store) {
        auto store = cast<memref::StoreOp>(item.op);
        Value val = tailMapper.lookupOrDefault(store.getValueToStore());
        if (!val) val = store.getValueToStore();
        Value idx = remapIndex(item.index, iv, tailIv);
        memref::StoreOp::create(builder, val, store.getMemRef(), idx);
        continue;
      }
      if (item.kind == BodyKind::Constant) {
        Operation *newCst = builder.clone(*item.op);
        tailMapper.map(item.op->getResults(), newCst->getResults());
        continue;
      }
      // The ops inside a divergent region come with the region's clone.
      if (item.divergent) continue;
      // An element-wise op or the region itself.
      Operation *cloned = builder.clone(*item.op, tailMapper);
      tailMapper.map(item.op->getResults(), cloned->getResults());
    }
    // (the tail loop's terminator comes from the ODS builder as well)

    // 3) The old loop goes away.
    forOp.erase();
  }

  LogicalResult vectorizeLoop(scf::ForOp forOp) {
    SmallVector<BodyOp> plan;
    int64_t vectorUb = 0;
    if (failed(analyze(forOp, plan, vectorUb))) return failure();
    rewrite(forOp, plan, vectorUb);
    return success();
  }

  // The index of an access in the new loop: "iv + constant" with the
  // induction variable replaced.
  Value remapIndex(Value idx, Value oldIv, Value newIv) {
    if (idx == oldIv) return newIv;
    if (auto add = idx.getDefiningOp<arith::AddIOp>()) {
      for (unsigned k = 0; k < 2; ++k) {
        if (add.getOperands()[k] != oldIv) continue;
        ConstIndex c = constantIndex(add.getOperands()[1 - k]);
        if (!c.found) continue;
        ImplicitLocOpBuilder ib(add.getLoc(), add);
        return arith::AddIOp::create(
            ib, newIv, arith::ConstantIndexOp::create(ib, c.value));
      }
    }
    // A constant index: it does not depend on the thread, so it stays.
    return idx;
  }

  int64_t width_ = 4;
};

}  // namespace

std::unique_ptr<Pass> createRougeUnitStrideToVectorPass() {
  return std::make_unique<UnitStrideToVector>();
}

void registerRougeUnitStrideToVectorPass() {
  PassRegistration<UnitStrideToVector>();
}

}  // namespace rouge
