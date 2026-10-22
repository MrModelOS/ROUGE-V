//===- SimtToScfImpl.h - the shared SIMT -> SCF lowering -*- C++ -*-===//
//
// The lowering of a SIMT kernel into explicit loops, written ONCE and used
// for two dialects:
//
//   * the upstream `gpu` dialect (the contour works with it today), and
//   * our own `rouge` dialect (the 2.0 plan: the kernel is described in a
//     dialect we own, and the SIMT-specific ops live in one place).
//
// The two dialects describe the same thing — a launch over a grid of blocks,
// the thread/block identity, a barrier and the memory accesses — so the
// lowering is the same transformation; only the op names differ. Duplicating
// the transformation would mean two places to fix the same bug, and the bugs
// here are subtle (dangling references to the erased kernel, the dominance
// of a value defined in the first phase, the induction variable's type).
//
// The transformation (the classic "SIMT to SIMD" phase decomposition):
//
//   launch %grid, %block, @k(%A)
//   @k(%A) { body0; barrier; body1 }
//
// becomes
//
//   scf.for %b = 0 to %grid.x {            // the launch: one block at a time
//     scf.for %t = 0 to %block.x { body0' }   // phase 0: no barrier inside
//     scf.for %t = 0 to %block.x { body1' }   // phase 1
//   }
//
// The barrier itself disappears: it only said "all the threads of the block
// reach this point", and after the decomposition every thread of phase 0 has
// run before phase 1 starts — the barrier's ordering requirement holds by
// construction.
//
// Splitting at the barriers is not optional. If the body stayed one loop,
// thread 0 would run the whole body (including the code after the barrier)
// before thread 1 ran the code before it — the barrier's meaning would be
// lost.
//
// What this lowering does NOT do (and says so):
//   * it does not vectorize — that is a separate pass;
//   * a divergent branch is NOT refused: our lowering runs the threads
//     SEQUENTIALLY, so each thread takes its own path — the same semantics
//     the hardware has, minus the parallelism. What that changes is the
//     memory ORDER between threads, and without a barrier that order is
//     undefined on a GPU anyway;
//   * a BARRIER inside a divergent branch IS refused: the barrier contract
//     requires every thread of the scope to reach the same instance of it,
//     so the result would be undefined, and the phase decomposition would
//     not hold;
//   * a kernel that uses SHARED memory is refused: the per-block copy of the
//     shared memory would be lost silently. The address space is part of the
//     memref type, so the CPU lowering CAN turn shared into private storage
//     — but only once the copy is there.
//
//===----------------------------------------------------------------------===//

#ifndef ROUGE_TRANSFORMS_SIMTTOSCFIMPL_H
#define ROUGE_TRANSFORMS_SIMTTOSCFIMPL_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace rouge {

using namespace mlir;  // NOLINT: the implementation header is internal

// Index cast without the deprecated builder::create overloads: inside the
// scf.for body callback we only have a plain OpBuilder, and MLIR 23's static
// create() wants an ImplicitLocOpBuilder (which cannot be built from an
// OpBuilder — the class is not copyable).
static Value makeIndexCast(OpBuilder &b, Location loc, Type type, Value v) {
  OperationState st(loc, arith::IndexCastOp::getOperationName());
  st.addOperands(v);
  st.addTypes(type);
  return b.create(st)->getResult(0);
}

// Cast to index only when it is actually needed: an index->index cast is
// invalid IR.
static Value castToIndex(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex()) return v;
  return makeIndexCast(b, loc, b.getIndexType(), v);
}

// The lowering itself. The op types are the template parameters, so the
// gpu-dialect and the rouge-dialect passes share the code.
template <typename KernelOp, typename LaunchOp, typename ThreadIdOp,
          typename BlockIdOp, typename BlockDimOp, typename GridDimOp,
          typename BarrierOp, typename ReturnOp, typename DimensionT>
struct SimtLoweringBase {
  // True if the dimension is the x one. Both dialects number the dimensions
  // x, y, z — and the check goes through the stringifier, so it does not
  // depend on the numbering.
  static bool isX(DimensionT dim) {
    return stringifyDimension(dim) == "x";
  }

  // True if the value (transitively, through the arithmetic the kernel uses)
  // depends on the thread id.
  static bool dependsOnThreadId(Value v, unsigned depth = 0) {
    if (!v || depth > 8) return false;
    if (isa<ThreadIdOp>(v.getDefiningOp())) return true;
    Operation *def = v.getDefiningOp();
    if (!def) return false;
    for (Value operand : def->getOperands())
      if (dependsOnThreadId(operand, depth + 1)) return true;
    return false;
  }

  // A barrier inside a conditional region breaks the barrier contract: the
  // threads of the scope do not all reach the same instance of it, so the
  // result is undefined and the phase decomposition does not hold.
  static bool hasBarrierInDivergentRegion(Operation *body) {
    bool found = false;
    body->walk([&](Operation *op) {
      if (found) return;
      if (!isa<BarrierOp>(op)) return;
      for (Operation *parent = op->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (isa<scf::IfOp>(parent)) {
          found = true;
          return;
        }
        if (isa<func::FuncOp>(parent)) break;
      }
    });
    return found;
  }

  // A kernel that already contains a loop (scf.while) cannot be phase-
  // decomposed by this pass: the barrier split assumes a flat instruction
  // stream. Refuse gracefully instead of crashing.
  static bool hasNestedLoop(Operation *kernel) {
    bool found = false;
    kernel->walk([&](scf::WhileOp op) { found = true; });
    return found;
  }

  // The kernels that use SHARED memory (a memref with a non-default memory
  // space) are refused: the per-block copy of the shared memory would be
  // lost silently, and the accesses would read the wrong memory.
  static bool usesSharedMemory(Operation *kernel) {
    bool hasShared = false;
    // The default memory space is whatever a memref without a space gets;
    // comparing against it avoids depending on how MLIR spells it.
    const Attribute defaultSpace =
        MemRefType::get({1}, Float32Type::get(kernel->getContext()))
            .getMemorySpace();
    kernel->walk([&](Operation *op) {
      auto check = [&](Type t) {
        if (auto mr = dyn_cast<MemRefType>(t))
          if (mr.getMemorySpace() != defaultSpace) hasShared = true;
      };
      for (Type t : op->getOperandTypes()) check(t);
      for (Type t : op->getResultTypes()) check(t);
    });
    return hasShared;
  }

  // True if the op READS or WRITES memory. An allocation is not a read: it
  // creates the memory, so it can be hoisted out of the phase loops (the
  // allocation is per block, and it must dominate every phase).
  static bool readsOrWritesMemory(Operation *op) {
    if (op->getNumRegions() != 0) return true;
    bool touches = false;
    if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      iface.getEffects(effects);
      for (const MemoryEffects::EffectInstance &eff : effects)
        if (isa<MemoryEffects::Read, MemoryEffects::Write>(eff.getEffect()))
          touches = true;
    }
    return touches;
  }

  // The kernel of a launch.
  static KernelOp findKernel(ModuleOp module, LaunchOp launch) {
    auto kernel = module.lookupSymbol<KernelOp>(
        launch.getKernelAttr().getLeafReference());
    if (!kernel)
      launch.emitError("the launch's kernel symbol does not name a kernel");
    return kernel;
  }

  // The default memory space: whatever a memref without a space gets;
  // comparing against it avoids depending on how MLIR spells it.
  static Attribute defaultMemorySpace(MLIRContext *ctx) {
    return MemRefType::get({1}, Float32Type::get(ctx)).getMemorySpace();
  }

  static bool isSharedMemRef(Type t, MLIRContext *ctx) {
    auto mr = dyn_cast<MemRefType>(t);
    return mr && mr.getMemorySpace() != defaultMemorySpace(ctx);
  }

  // The SHARED memory of a kernel.
  //
  // In our dialect the shared memory is DECLARED inside the kernel (a
  // memref.alloc with the shared address space) — exactly like CUDA's
  // ".shared" declaration. The kernel runs once per block, so turning the
  // declaration into a private allocation gives exactly the lifetime the
  // shared memory has: it lives as long as the block, and the next block
  // gets a fresh one. The barrier keeps the phases apart, so the ordering
  // requirement of the shared memory holds.
  //
  // A shared memory passed as a PARAMETER cannot be handled this way: the
  // per-block copy would have to live in the caller, and the launch does not
  // know the size. That is refused loudly.
  static LogicalResult lowerSharedMemory(KernelOp kernel) {
    MLIRContext *ctx = kernel.getContext();
    Attribute defaultSpace = defaultMemorySpace(ctx);

    for (Type t : kernel.getFunctionType().getInputs()) {
      if (!isSharedMemRef(t, ctx)) continue;
      kernel.emitError(
          "a shared memory passed as a parameter is refused: declare it "
          "inside the kernel (a memref.alloc with the shared address space) "
          "so that the per-block copy is created together with the block");
      return failure();
    }

    SmallVector<Operation *> allocs;
    kernel.walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp>(op)) allocs.push_back(op);
    });
    SmallVector<Value> sharedAllocs;
    for (Operation *op : allocs) {
      auto mr = cast<MemRefType>(op->getResult(0).getType());
      if (mr.getMemorySpace() == defaultSpace) continue;
      // MLIR 23: the layout comes before the memory space.
      MemRefType newType = MemRefType::get(
          mr.getShape(), mr.getElementType(), mr.getLayout(), defaultSpace);
      // The type of the result is changed in place: the op keeps its
      // attributes (the alignment, the dynamic sizes) and its uses (the
      // dealloc, the loads) follow the new type.
      op->getResult(0).setType(newType);
      sharedAllocs.push_back(op->getResult(0));
    }

    // The shared memory dies with the BLOCK, so there is nothing to free:
    // the dealloc is removed. Left in place, it would run once per thread
    // (the phases are loops over the threads) — a double free.
    SmallVector<Operation *> deallocs;
    kernel.walk([&](Operation *op) {
      if (!isa<memref::DeallocOp>(op)) return;
      for (Value operand : op->getOperands())
        if (llvm::is_contained(sharedAllocs, operand)) {
          deallocs.push_back(op);
          return;
        }
    });
    for (Operation *op : deallocs) op->erase();
    return success();
  }

  // The kernel body: phases of thread loops. `launch` may be a null op: a
  // kernel without a launch is lowered on its own (the block size then comes
  // from an explicit block_dim in the body). `takesBlockId` reports whether
  // the new function takes the block id as an extra argument — the caller
  // needs it to build the call.
  static LogicalResult convertKernel(KernelOp kernel, LaunchOp launch,
                                     bool &takesBlockId) {
    Region &body = kernel.getBody();
    if (body.empty()) return success();

    // The shared memory: the declaration inside the kernel becomes a private
    // allocation (the kernel runs once per block, so the allocation is per
    // block). A shared memory that arrives as a parameter is refused.
    if (failed(lowerSharedMemory(kernel))) return failure();

    if (usesSharedMemory(kernel)) {
      kernel.emitError("a kernel using shared memory is refused for now (the "
                       "per-block copy of the shared memory would be lost "
                       "silently; the shared memory must become a memref with "
                       "a memory space and be copied per block)");
      return failure();
    }

    if (hasBarrierInDivergentRegion(kernel)) {
      kernel.emitError("a barrier inside a divergent branch is refused — "
                       "every thread of the scope must reach the same "
                       "instance of the barrier, so the result would be "
                       "undefined");
      return failure();
    }

    Location loc = kernel.getLoc();
    OpBuilder typeBuilder(kernel.getContext());

    Block &srcBlock = body.front();
    SmallVector<Operation *> ops;
    for (Operation &op : srcBlock)
      if (!isa<ReturnOp>(op)) ops.push_back(&op);

    // Phase split at the barriers. The phases hold INDICES into `ops`: the
    // ops themselves are cloned into the new function below, and the CLONES
    // are what end up in the loops. (Cloning only the phase ops would leave
    // the clones referring to the values of the old kernel, which is erased
    // at the end — a dangling reference.)
    SmallVector<SmallVector<unsigned>> phases;
    SmallVector<unsigned> current;
    for (unsigned i = 0; i < ops.size(); ++i) {
      if (isa<BarrierOp>(ops[i])) {
        phases.push_back(current);
        current.clear();
        continue;
      }
      current.push_back(i);
    }
    phases.push_back(current);

    // The thread count: the launch's block size, or an explicit block_dim x
    // in the body.
    Value threadCount;
    Type threadIdType = typeBuilder.getIndexType();
    for (Operation &op : srcBlock) {
      if (auto tid = dyn_cast<ThreadIdOp>(&op)) {
        threadIdType = tid.getResult().getType();
        break;
      }
    }
    if (launch) {
      threadCount = launch.getBlockSizeX();
    } else {
      for (Operation &op : srcBlock)
        if (auto dim = dyn_cast<BlockDimOp>(&op))
          if (isX(dim.getDimension())) threadCount = dim.getResult();
    }
    if (!threadCount) {
      kernel.emitError("no block size (the launch's block size or an explicit "
                       "block_dim x) — cannot size the thread loop");
      return failure();
    }

    // The body's ops split into two groups:
    //   * loop-invariant (constants, address computations that do not depend
    //     on the thread id) — they are hoisted OUT of the thread loops, once
    //     per function; otherwise a value defined in the first phase would
    //     not dominate its use in the second one;
    //   * thread-dependent — they go into the body of the phase's loop.
    // Both groups are CLONED: the originals die with the old kernel, and a
    // clone that still referred to them would be a dangling reference.
    std::function<bool(Operation *, unsigned)> dependsOnThread =
        [&](Operation *op, unsigned depth) -> bool {
      if (depth > 16) return true;  // unknown -> assume dependent (safe)
      // The thread id itself is the loop's induction variable: it is
      // "dependent" by definition (hoisting it out of the loop would give
      // every thread the same value).
      if (isa<ThreadIdOp>(op)) return true;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def) continue;
        if (isa<ThreadIdOp>(def)) return true;
        if (dependsOnThread(def, depth + 1)) return true;
      }
      // The NESTED regions: a loop or a divergent branch uses the values of
      // its body, so an op that owns a region depends on the thread id if
      // anything inside the region does.
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &inner : block)
            if (dependsOnThread(&inner, depth + 1)) return true;
      return false;
    };

    SmallVector<Operation *> invariantOps, dependentOps;
    for (Operation *op : ops) {
      if (isa<BarrierOp>(op)) continue;
      // The identity and the sizes are not cloned: they are mapped to the
      // induction variable, the block argument and the bounds of the loops
      // (see below).
      if (isa<ThreadIdOp, BlockIdOp, BlockDimOp, GridDimOp>(op)) continue;
      // A MEMORY access stays in its phase. Hoisting it out of the phase
      // loops would read the memory at the wrong time: another phase may
      // have written it in between (an atomic add in an earlier phase, for
      // example), and the hoisted read would then see the value from before
      // the write — a silent wrong answer, not a crash. The parameter read
      // is the exception: the parameter buffer is immutable.
      const bool touchesMemory = readsOrWritesMemory(op);
      const bool isParamRead = StringRef(op->getName().getStringRef())
                                   .starts_with("ld.param.");
      if (dependsOnThread(op, 0) || (touchesMemory && !isParamRead))
        dependentOps.push_back(op);
      else
        invariantOps.push_back(op);
    }

    // A third group: the values that depend on the thread id but ONLY through
    // pure arithmetic — `index_cast(tid)`, `tid * 4`, `tid + 1`. The phases
    // are separate loops, so such a value does not survive from one loop to
    // the next; but it does not have to: it is a pure function of the
    // induction variable, so every phase that needs it can RECOMPUTE it.
    //
    // This is not an optimisation, it is what real kernels look like: nvcc
    // computes the thread index once and uses it in every phase, and
    // refusing that would refuse most real kernels for no reason. What is
    // still refused is a value that really crosses the boundary — one that
    // depends on MEMORY (`index_cast(load ...)`), because recomputing it
    // would silently read the memory at a different time.
    DenseSet<Operation *> invariantSet(invariantOps.begin(),
                                       invariantOps.end());
    DenseSet<Operation *> rematSet;
    std::function<bool(Operation *)> isRematerialisable =
        [&](Operation *op) -> bool {
      if (invariantSet.count(op) || rematSet.count(op)) return true;
      if (op->getNumRegions() != 0) return false;
      if (!isMemoryEffectFree(op)) return false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def) return false;  // a block argument: not ours to recompute
        if (isa<ThreadIdOp, BlockIdOp, BlockDimOp, GridDimOp>(def)) continue;
        if (invariantSet.count(def) || rematSet.count(def)) continue;
        if (!isRematerialisable(def)) return false;
      }
      rematSet.insert(op);
      return true;
    };
    for (Operation *op : dependentOps) isRematerialisable(op);

    // The position of every op in the original body: the recomputed values
    // are cloned in the original program order, so an op always comes after
    // the values it uses.
    DenseMap<Operation *, unsigned> opIndex;
    for (unsigned i = 0; i < ops.size(); ++i) opIndex[ops[i]] = i;

    // The new function: the same arguments plus the block id (if the body
    // uses it), because the block id now lives in the launch's loop.
    auto fnType = cast<FunctionType>(kernel.getFunctionType());
    SmallVector<Type> newArgTypes(fnType.getInputs());
    bool usesBlockId = false;
    for (Operation &op : srcBlock)
      if (isa<BlockIdOp>(&op)) usesBlockId = true;
    if (usesBlockId) newArgTypes.push_back(threadIdType);
    takesBlockId = usesBlockId;

    ModuleOp module = kernel->template getParentOfType<ModuleOp>();
    const std::string tmpName = (kernel.getName() + ".rouge.tmp").str();
    ImplicitLocOpBuilder moduleBuilder(loc, module.getBodyRegion());
    moduleBuilder.setInsertionPointToStart(module.getBody());
    auto newFn = func::FuncOp::create(moduleBuilder, tmpName,
                                      typeBuilder.getFunctionType(newArgTypes, {}));
    newFn.setVisibility(kernel.getVisibility());
    Block *entry = newFn.addEntryBlock();
    ImplicitLocOpBuilder builder(loc, newFn.getRegion());
    builder.setInsertionPointToStart(entry);

    // The old kernel's arguments and the new function's arguments must be
    // the same values: the ops cloned below reference the old ones, and the
    // old kernel is erased at the end.
    {
      Block &oldEntry = body.front();
      auto oldArgs = oldEntry.getArguments();
      auto newArgs = entry->getArguments();
      for (unsigned i = 0; i < oldArgs.size() && i < newArgs.size(); ++i)
        oldArgs[i].replaceAllUsesWith(newArgs[i]);
    }
    Value blockIdArg =
        usesBlockId ? entry->getArgument(newArgTypes.size() - 1) : Value();

    // The thread count usually comes from the launch op (an operand of it,
    // defined in the HOST function). The new kernel function is a sibling of
    // that function, not a nested region, so the value is not visible there:
    // it has to be cloned in — otherwise the loop bound would dangle the
    // moment the launch op is erased.
    if (threadCount && !isa<BlockArgument>(threadCount)) {
      if (Operation *def = threadCount.getDefiningOp()) {
        Operation *clone = builder.clone(*def);
        threadCount =
            clone->getResult(cast<OpResult>(threadCount).getResultNumber());
      }
    }
    // The identity ops (the thread id, the block id) are not tied to a
    // phase: they are values of the whole kernel, and a LATER phase may use
    // the thread id that an earlier phase defined. So their mapping is set up
    // in EVERY phase — otherwise the clone of a later phase would refer to
    // the op in the old kernel, which is erased at the end (a dangling
    // reference, and a crash much later).
    SmallVector<Operation *> threadIdOps, blockIdOps, blockDimOps, gridDimOps;
    kernel.getBody().walk([&](Operation *op) {
      if (isa<ThreadIdOp>(op))
        threadIdOps.push_back(op);
      else if (isa<BlockIdOp>(op))
        blockIdOps.push_back(op);
      else if (isa<BlockDimOp>(op))
        blockDimOps.push_back(op);
      else if (isa<GridDimOp>(op))
        gridDimOps.push_back(op);
    });

    Value threadBound = castToIndex(builder, loc, threadCount);
    Value c0 = arith::ConstantIndexOp::create(builder, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, 1);

    // The grid size: the kernel may ask for it (rouge.grid_dim). It lives in
    // the host function (it is an operand of the launch), so it has to be
    // cloned in — exactly like the block size.
    Value gridBoundInKernel;
    if (!gridDimOps.empty()) {
      if (!launch)
        return failure();
      Value gridSize = launch.getGridSizeX();
      if (gridSize && !isa<BlockArgument>(gridSize)) {
        if (Operation *def = gridSize.getDefiningOp()) {
          Operation *clone = builder.clone(*def);
          gridSize =
              clone->getResult(cast<OpResult>(gridSize).getResultNumber());
        }
      }
      gridBoundInKernel = castToIndex(builder, loc, gridSize);
    }

    // Hoisted invariants: cloned into the function's entry block, in order.
    // The clone goes through an IRMapping, so an invariant op that uses the
    // result of another invariant op (a constant used by an address
    // computation) refers to the CLONE, and the mapping also covers the
    // contents of nested regions.
    IRMapping invMapper;
    // The block id and the sizes are mapped here as well: an invariant that
    // uses the block id (the global index, for example) must refer to the
    // argument of the new function, not to the op of the old kernel — that
    // one is erased at the end (a dangling reference).
    for (Operation *op : blockIdOps)
      if (blockIdArg) invMapper.map(op->getResult(0), blockIdArg);
    for (Operation *op : blockDimOps)
      invMapper.map(op->getResult(0), threadBound);
    for (Operation *op : gridDimOps)
      invMapper.map(op->getResult(0), gridBoundInKernel);
    DenseMap<Operation *, Operation *> invariantClone;
    for (Operation *op : invariantOps) {
      Operation *clone = builder.clone(*op, invMapper);
      invMapper.map(op->getResults(), clone->getResults());
      invariantClone[op] = clone;
    }

    // A value defined in one phase and used in another: the phases are
    // SEPARATE loops, so such a value does not survive from one loop to the
    // next. Carrying it would mean passing it between the loops (the
    // iter_args of scf.for), and until that is implemented the honest answer
    // is a refusal: the kernel is not silently mis-lowered.
    {
      DenseMap<Operation *, unsigned> phaseOf;
      for (unsigned p = 0; p < phases.size(); ++p)
        for (unsigned idx : phases[p]) phaseOf[ops[idx]] = p;
      for (unsigned p = 0; p < phases.size(); ++p) {
        for (unsigned idx : phases[p]) {
          Operation *op = ops[idx];
          if (isa<ThreadIdOp, BlockIdOp>(op)) continue;
          // The operands of the op AND of everything inside its regions: a
          // value used inside a nested region crosses the phase boundary
          // just as much as one used by the op itself.
          SmallVector<Operation *> inner;
          op->walk([&](Operation *innerOp) { inner.push_back(innerOp); });
          for (Operation *innerOp : inner)
            for (Value operand : innerOp->getOperands()) {
              Operation *def = operand.getDefiningOp();
              if (!def) continue;
              if (isa<ThreadIdOp, BlockIdOp>(def)) continue;
              // A pure function of the thread id is recomputed in every phase
              // that needs it (see rematSet above).
              if (rematSet.count(def)) continue;
              // The invariants are hoisted out of the loops, so they dominate
              // every phase and are not a cross-phase dependency.
              if (std::find(invariantOps.begin(), invariantOps.end(), def) !=
                  invariantOps.end())
                continue;
              auto it = phaseOf.find(def);
              if (it == phaseOf.end() || it->second == p) continue;
              op->emitError(
                  "a value defined in one phase is used in another phase — the "
                  "phases are separate loops, so the value does not survive "
                  "between them; carry it through the loop (iter_args) or "
                  "refuse the kernel");
              return failure();
            }
        }
      }
    }

    // Phases now refer to the dependent ops by index. The invariants are
    // hoisted out of the loops and the rematerialisable values are
    // recomputed inside each phase, so neither is bound to a phase.
    SmallVector<SmallVector<unsigned>> phaseDependent;
    for (SmallVector<unsigned> &phase : phases) {
      SmallVector<unsigned> dep;
      for (unsigned idx : phase) {
        Operation *op = ops[idx];
        if (std::find(invariantOps.begin(), invariantOps.end(), op) !=
            invariantOps.end())
          continue;
        if (rematSet.count(op)) continue;
        dep.push_back(idx);
      }
      phaseDependent.push_back(dep);
    }

    bool phaseFailed = false;
    auto buildPhase = [&](SmallVector<unsigned> &phase) -> LogicalResult {
      // The op is created through the raw OperationState: the ODS builder in
      // MLIR 23 leaves the body region to the caller, and the op must land in
      // a block before it can be touched at all.
      OperationState st(loc, scf::ForOp::getOperationName());
      st.addOperands({c0, threadBound, c1});
      st.addTypes({});
      st.addRegion();
      auto forOp = cast<scf::ForOp>(builder.create(st));
      Region &bodyRegion = forOp.getRegion();
      Block *bodyBlock = nullptr;
      if (bodyRegion.empty()) {
        bodyBlock = new Block();
        bodyRegion.push_back(bodyBlock);
      } else {
        bodyBlock = &bodyRegion.front();
      }
      if (bodyBlock->getNumArguments() == 0)
        bodyBlock->addArgument(builder.getIndexType(), loc);
      Value iv = bodyBlock->getArgument(0);

      OpBuilder bodyBuilder = builder;
      bodyBuilder.setInsertionPointToStart(bodyBlock);
      // The thread id becomes the induction variable; if the op's type in
      // this MLIR version differs, an explicit cast keeps the uses valid.
      Value tidValue = iv;
      if (tidValue.getType() != threadIdType)
        tidValue = makeIndexCast(bodyBuilder, loc, threadIdType, tidValue);
      // Clone the phase through an IRMapping. The mapping carries three
      // things: the thread id -> the induction variable, the hoisted
      // invariants -> their clones, and the results of the ops of this phase
      // -> their clones. Going through the mapping (instead of rewriting the
      // operands by hand) is what makes the NESTED regions come along: a
      // divergent branch inside the body carries a region, and its contents
      // refer to the values of the body — without the mapping they would
      // keep referring to the old kernel, which is a dangling reference.
      IRMapping mapper;
      // The identity ops are mapped in EVERY phase (see above): a later
      // phase may use the thread id that an earlier phase defined.
      for (Operation *op : threadIdOps) {
        auto tid = cast<ThreadIdOp>(op);
        if (!isX(tid.getDimension())) {
          tid.emitError("only the x dimension of the thread id is supported");
          phaseFailed = true;
          return failure();
        }
        mapper.map(tid.getResult(), tidValue);
      }
      for (Operation *op : blockIdOps) {
        auto bid = cast<BlockIdOp>(op);
        if (!isX(bid.getDimension())) {
          bid.emitError("only the x dimension of the block id is supported");
          phaseFailed = true;
          return failure();
        }
        if (blockIdArg) mapper.map(bid.getResult(), blockIdArg);
      }
      // The sizes: the bounds of the loops.
      for (Operation *op : blockDimOps) {
        auto dim = cast<BlockDimOp>(op);
        if (!isX(dim.getDimension())) {
          dim.emitError("only the x dimension of the block size is supported");
          phaseFailed = true;
          return failure();
        }
        mapper.map(dim.getResult(), threadBound);
      }
      for (Operation *op : gridDimOps) {
        auto dim = cast<GridDimOp>(op);
        if (!isX(dim.getDimension())) {
          dim.emitError("only the x dimension of the grid size is supported");
          phaseFailed = true;
          return failure();
        }
        mapper.map(dim.getResult(), gridBoundInKernel);
      }
      // The invariants are visible from the loop body.
      for (Operation *op : invariantOps) {
        Operation *clone = invariantClone[op];
        for (auto pair : llvm::zip(op->getResults(), clone->getResults()))
          mapper.map(std::get<0>(pair), std::get<1>(pair));
      }
      // The values this phase needs and can recompute (a pure function of
      // the thread id): collected transitively from the phase's own operands
      // and cloned in the original program order, so an operand is always
      // cloned before its use.
      {
        SmallVector<Operation *> needed;
        SmallVector<Operation *> work;
        DenseSet<Operation *> seen;
        for (unsigned idx : phase) {
          // The operands of the op AND of everything inside its regions: a
          // nested region (a loop, a divergent branch) uses the values of
          // the body, and those values must be remapped too — otherwise the
          // clone would refer to the old kernel, which is erased at the end.
          SmallVector<Operation *> inner;
          ops[idx]->walk([&](Operation *op) { inner.push_back(op); });
          for (Operation *innerOp : inner)
            for (Value operand : innerOp->getOperands())
              if (Operation *def = operand.getDefiningOp()) work.push_back(def);
        }
        while (!work.empty()) {
          Operation *op = work.pop_back_val();
          if (!seen.insert(op).second) continue;
          if (!rematSet.count(op)) continue;
          needed.push_back(op);
          for (Value operand : op->getOperands())
            if (Operation *def = operand.getDefiningOp()) work.push_back(def);
        }
        std::sort(needed.begin(), needed.end(), [&](Operation *a, Operation *b) {
          return opIndex.lookup(a) < opIndex.lookup(b);
        });
        for (Operation *op : needed) {
          Operation *cloned = bodyBuilder.clone(*op, mapper);
          mapper.map(op->getResults(), cloned->getResults());
        }
      }
      for (unsigned idx : phase) {
        Operation *op = ops[idx];
        // The thread id, the block id and the sizes have no counterpart in
        // the lowered form: they ARE the induction variable, the block
        // argument and the bounds of the loops, so they are not cloned at
        // all (cloning them would also overwrite the mapping).
        if (isa<ThreadIdOp, BlockIdOp, BlockDimOp, GridDimOp>(op)) continue;
        Operation *cloned = bodyBuilder.clone(*op, mapper);
        mapper.map(op->getResults(), cloned->getResults());
      }
      // The body of an scf.for must end with scf.yield. The op is created
      // through the dialect's own builder (a raw OperationState would need
      // the dialect to be loaded, and the check for that is what the MLIR
      // context does on creation).
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(bodyBlock);
        scf::YieldOp::create(builder);
      }
      builder.setInsertionPointAfter(forOp);
      return phaseFailed ? failure() : success();
    };

    {
      OpBuilder::InsertionGuard guard(builder);
      for (SmallVector<unsigned> &phase : phaseDependent)
        if (failed(buildPhase(phase))) return failure();
    }
    func::ReturnOp::create(builder);

    const std::string realName = kernel.getName().str();
    kernel->erase();
    newFn.setName(realName);
    return success();
  }

  // The launch: a loop over the grid that calls the (now loop-based) kernel.
  static LogicalResult convertLaunch(LaunchOp launch, bool takesBlockId) {
    Location loc = launch.getLoc();
    ModuleOp module = launch->template getParentOfType<ModuleOp>();
    ImplicitLocOpBuilder builder(loc, module.getBodyRegion());
    builder.setInsertionPoint(launch);

    Value gridX = launch.getGridSizeX();
    Value c0 = arith::ConstantIndexOp::create(builder, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, 1);
    Value gridBound = castToIndex(builder, loc, gridX);

    OperationState gst(loc, scf::ForOp::getOperationName());
    gst.addOperands({c0, gridBound, c1});
    gst.addTypes({});
    gst.addRegion();
    auto forOp = cast<scf::ForOp>(builder.create(gst));
    // The body of the loop must EXIST before anyone inserts into it: the op
    // was built from a raw OperationState, so its region is empty until we
    // push a block into it. (Taking getBody() of an empty region and
    // inserting into it is how the module ended up with a destroyed op still
    // sitting in a block.)
    Region &bodyRegion = forOp.getRegion();
    Block *bodyBlock = new Block();
    bodyBlock->addArgument(builder.getIndexType(), loc);
    bodyRegion.push_back(bodyBlock);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);

    // The call: the kernel operands, plus the block id ONLY if the kernel
    // takes one (convertKernel appends it to the signature exactly when the
    // body uses the block id; passing it unconditionally would make the call
    // have an extra operand of the wrong type).
    SmallVector<Value> args(launch.getKernelOperands().begin(),
                            launch.getKernelOperands().end());
    if (takesBlockId) {
      // The block id argument is appended to the kernel signature, so its
      // type is the one the kernel expects.
      Type blockIdType = builder.getIndexType();
      auto kernel = module.lookupSymbol<KernelOp>(
          launch.getKernelAttr().getLeafReference());
      if (kernel && !kernel.getFunctionType().getInputs().empty())
        blockIdType = kernel.getFunctionType().getInputs().back();
      Value blockId = forOp.getInductionVar();
      if (blockId.getType() != blockIdType)
        blockId = makeIndexCast(builder, loc, blockIdType, blockId);
      args.push_back(blockId);
    }

    func::CallOp::create(builder, launch.getKernelAttr().getLeafReference(),
                         launch->getResultTypes(), args);
    // The body of an scf.for must end with scf.yield.
    {
      OpBuilder::InsertionGuard yieldGuard(builder);
      builder.setInsertionPointToEnd(bodyBlock);
      scf::YieldOp::create(builder);
    }

    // The launch op itself goes away.
    launch->erase();
    return success();
  }
};

}  // namespace rouge

#endif // ROUGE_TRANSFORMS_SIMTTOSCFIMPL_H
