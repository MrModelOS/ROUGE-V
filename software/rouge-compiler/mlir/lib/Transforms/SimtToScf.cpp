//=- ROUGE-V: SIMT -> SCF (the first rewriting pass of the MLIR contour).
//
// ВНИМАНИЕ: проход написан, но НЕ включён в сборку (см. mlir/CMakeLists.txt
// и раздел «SIMT -> SCF: что нашли» в mlir/README.md). Он переводит ядро в
// форму с явными циклами и разбивкой на фазы по барьерам, но в MLIR 23
// обходит ещё несколько граблей (список — в README). Включать раньше времени
// нельзя: контур обязан оставаться зелёным.
//
//=.
//
// Остальной комментарий —
//
// Stage 2 of the pipeline (docs/06, mlir/README.md) turns a GPU kernel into
// vector code for RVV. The first step is to make the SIMT structure explicit:
// a thread is an iteration of a loop, not a magic identity.
//
// The transformation (the classic "SIMT to SIMD" phase decomposition):
//
//   gpu.launch_func %grid, %block, @k(%A)
//   gpu.func @k(%A) kernel { body0; gpu.barrier; body1 }
//
// becomes
//
//   scf.for %b = 0 to %grid.x {            // the launch: one block at a time
//     scf.for %t = 0 to %block.x { body0' }   // phase 0: no barrier inside
//     scf.for %t = 0 to %block.x { body1' }   // phase 1
//   }
//
// where gpu.thread_id x becomes the loop induction variable and
// gpu.block_id x becomes the launch loop's variable (passed to the kernel as
// an extra argument). The barrier itself disappears: it only said "all
// threads of the block reach this point", and after the decomposition every
// thread of phase 0 has run before phase 1 starts — the barrier's ordering
// requirement holds by construction.
//
// Splitting at barriers is not optional. If the body stayed one loop, thread
// 0 would run the whole body (including the code after the barrier) before
// thread 1 ran the code before it — the barrier's meaning would be lost.
// Phases keep it.
//
// What this pass does NOT do (and says so):
//   * it does not vectorize — that is rouge-unit-stride-to-vector;
//   * it does not model divergent control flow: an if whose condition
//     depends on the thread id is REFUSED with a loud error, not silently
//     mis-lowered;
//   * only the x dimension of the thread/block/grid ids is handled; the y and
//     z dimensions are refused loudly;
//   * shared memory stays memref accesses — the phase decomposition is what
//     makes a later shared-memory lowering sound.
//
//   rouge-opt --rouge-simt-to-scf kernel.mlir
//
//=.

#include "rouge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <map>
#include <string>
#include <memory>
#include <string>

using namespace mlir;

namespace rouge {
namespace {

// True if the value (transitively, through the arithmetic the kernel uses)
// depends on the thread id. Such a value inside a branch condition means
// divergence, which this pass refuses.
bool dependsOnThreadId(Value v, unsigned depth = 0) {
  if (!v || depth > 8) return false;
  if (isa<gpu::ThreadIdOp>(v.getDefiningOp())) return true;
  Operation *def = v.getDefiningOp();
  if (!def) return false;
  for (Value operand : def->getOperands())
    if (dependsOnThreadId(operand, depth + 1)) return true;
  return false;
}

// A branch whose condition depends on the thread id diverges.
bool hasDivergentBranch(Operation *body) {
  bool found = false;
  body->walk([&](Operation *op) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (dependsOnThreadId(ifOp.getCondition())) found = true;
    }
    if (auto br = dyn_cast<cf::CondBranchOp>(op)) {
      if (dependsOnThreadId(br.getCondition())) found = true;
    }
  });
  return found;
}

// Index cast without the deprecated builder::create overloads: inside the
// scf.for body callback we only have a plain OpBuilder, and MLIR 23's static
// create() wants an ImplicitLocOpBuilder (which cannot be built from an
// OpBuilder — the class is not copyable).
// scf.yield terminator (MLIR 23 wants an ImplicitLocOpBuilder, which cannot
// be made from a plain OpBuilder, so the op is built directly).
static void makeYield(OpBuilder &b, Location loc) {
  OperationState st(loc, scf::YieldOp::getOperationName());
  st.addOperands({});
  st.addTypes({});
  b.create(st);
}

static Value makeIndexCast(OpBuilder &b, Location loc, Type type, Value v) {
  OperationState st(loc, arith::IndexCastOp::getOperationName());
  st.addOperands(v);
  st.addTypes(type);
  return b.create(st)->getResult(0);
}

// Cast to index only when it is actually needed: the gpu dialect's grid and
// block sizes are already index, and an index->index cast is invalid IR.
static Value castToIndex(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex()) return v;
  return makeIndexCast(b, loc, b.getIndexType(), v);
}


// Перемапливает операнды клона: thread id -> индуктивная переменная,
// определение из инвариантов -> их клон, определение из той же фазы -> её
// клон. Всё, что осталось, — аргумент функции (уже переподставлен) или
// литерал.
//
// Это НЕЛЬЗЯ пропускать: клон, ссылающийся на оригинал, — это висячая
// ссылка (оригинал умирает вместе со старым ядром), и она проявляется
// сильно позже и совсем в другом месте.
struct SimtToScf
    : public PassWrapper<SimtToScf, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimtToScf)

  // Ядра, принимающие block id дополнительным аргументом: без этого
  // вызов получает лишний операнд не того типа.
  std::map<std::string, bool> kernelTakesBlockId_;

  StringRef getArgument() const final { return "rouge-simt-to-scf"; }
  StringRef getDescription() const final {
    return "Turn SIMT thread/block ids into explicit loops (phase "
           "decomposition at barriers)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.getContext()->disableMultithreading();
    // The dialects whose ops this pass creates must be LOADED in the context:
    // an input without scf ops never loads the SCF dialect, and creating an
    // op of an unloaded dialect is refused ("created with unregistered
    // dialect").
    MLIRContext *ctx = module.getContext();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    if (failed(verify(module))) {
      module.emitError("rouge-simt-to-scf: module verification failed");
      return;
    }

    // The launch that calls each kernel (if any): the launch owns the grid
    // and block sizes, the kernel owns the body.
    SmallVector<gpu::GPUFuncOp> kernels;
    module.walk([&](gpu::GPUFuncOp f) { kernels.push_back(f); });

    unsigned converted = 0, refused = 0;
    for (gpu::GPUFuncOp kernel : kernels) {
      gpu::LaunchFuncOp launch = findLaunch(module, kernel);
      if (failed(convertKernel(kernel, launch))) {
        ++refused;
        continue;
      }
      if (launch) {
        if (failed(convertLaunch(launch))) {
          ++refused;
          continue;
        }
      }
      ++converted;
    }
    llvm::errs() << "rouge-simt-to-scf: kernels converted=" << converted
                 << " refused=" << refused << "\n";

  }

  gpu::LaunchFuncOp findLaunch(ModuleOp module, gpu::GPUFuncOp kernel) {
    gpu::LaunchFuncOp found;
    module.walk([&](gpu::LaunchFuncOp launch) {
      if (found) return;
      auto callee = launch.getKernelAttr();
      if (callee.getLeafReference() == kernel.getName()) found = launch;
    });
    return found;
  }

  // The kernel body: phases of thread loops.
  LogicalResult convertKernel(gpu::GPUFuncOp kernel, gpu::LaunchFuncOp launch) {
    Region &body = kernel.getBody();
    if (body.empty()) return success();

    if (hasDivergentBranch(kernel)) {
      kernel.emitError("rouge-simt-to-scf: divergent control flow is not "
                       "supported yet (the pass refuses instead of guessing)");
      return failure();
    }

    Location loc = kernel.getLoc();
    OpBuilder typeBuilder(kernel.getContext());

    Block &srcBlock = body.front();
    SmallVector<Operation *> ops;
    for (Operation &op : srcBlock)
      if (!isa<gpu::ReturnOp>(op)) ops.push_back(&op);

    // Phase split at barriers. The phases hold INDICES into `ops`: the ops
    // themselves are cloned into the new function below, and the CLONES are
    // what end up in the loops. (Cloning only the phase ops would leave the
    // clones referring to values of the old kernel, which is erased at the
    // end — a dangling reference.)
    SmallVector<SmallVector<unsigned>> phases;
    SmallVector<unsigned> current;
    for (unsigned i = 0; i < ops.size(); ++i) {
      if (isa<gpu::BarrierOp>(ops[i])) {
        phases.push_back(current);
        current.clear();
        continue;
      }
      current.push_back(i);
    }
    phases.push_back(current);

    // The thread count: the launch's block size, or an explicit
    // gpu.block_dim x in the body.
    Value threadCount;
    Type threadIdType = typeBuilder.getIndexType();
    for (Operation &op : srcBlock) {
      if (auto tid = dyn_cast<gpu::ThreadIdOp>(&op)) {
        threadIdType = tid.getResult().getType();
        break;
      }
    }
    if (launch) {
      threadCount = launch.getBlockSizeX();
    } else {
      for (Operation &op : srcBlock)
        if (auto dim = dyn_cast<gpu::BlockDimOp>(&op))
          if (dim.getDimension() == gpu::Dimension::x)
            threadCount = dim.getResult();
    }
    if (!threadCount) {
      kernel.emitError("rouge-simt-to-scf: no block size (gpu.launch_func's "
                       "blockDim.x or an explicit gpu.block_dim x) — cannot "
                       "size the thread loop");
      return failure();
    }

    // The new function: the same arguments plus the block id (if the body
    // uses it), because the block id now lives in the launch's loop.
    auto fnType = cast<FunctionType>(kernel.getFunctionType());
    SmallVector<Type> newArgTypes(fnType.getInputs());
    bool usesBlockId = false;
    for (Operation &op : srcBlock)
      if (isa<gpu::BlockIdOp>(&op)) usesBlockId = true;
    if (usesBlockId) newArgTypes.push_back(threadIdType);
    kernelTakesBlockId_[kernel.getName().str()] = usesBlockId;

    ModuleOp module = kernel->getParentOfType<ModuleOp>();
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
    Value blockIdArg = usesBlockId ? entry->getArgument(newArgTypes.size() - 1)
                                   : Value();

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
    Value threadBound = castToIndex(builder, loc, threadCount);
    Value c0 = arith::ConstantIndexOp::create(builder, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, 1);

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
      if (isa<gpu::ThreadIdOp>(op)) return true;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def) continue;
        if (isa<gpu::ThreadIdOp>(def)) return true;
        if (dependsOnThread(def, depth + 1)) return true;
      }
      return false;
    };

    SmallVector<Operation *> invariantOps, dependentOps;
    for (Operation *op : ops) {
      if (isa<gpu::BarrierOp>(op)) continue;
      if (dependsOnThread(op, 0))
        dependentOps.push_back(op);
      else
        invariantOps.push_back(op);
    }

    // Hoisted invariants: cloned into the function's entry block, in order.
    // Their operands are remapped too: an invariant op may well use the
    // result of another invariant op (a constant used by an address
    // computation), and that result exists only as a clone.
    DenseMap<Operation *, Operation *> invariantClone;
    for (Operation *op : invariantOps)
      invariantClone[op] = builder.clone(*op);
    for (Operation *op : invariantOps) {
      Operation *cloned = invariantClone[op];
      SmallVector<Value> newOperands;
      for (Value operand : cloned->getOperands()) {
        Operation *def = operand.getDefiningOp();
        auto it = def ? invariantClone.find(def) : invariantClone.end();
        if (it != invariantClone.end() && it->second != cloned) {
          newOperands.push_back(
              it->second->getResult(cast<OpResult>(operand).getResultNumber()));
          continue;
        }
        newOperands.push_back(operand);
      }
      cloned->setOperands(newOperands);
    }

    // Phases now refer to the dependent ops by index.
    SmallVector<SmallVector<unsigned>> phaseDependent;
    for (SmallVector<unsigned> &phase : phases) {
      SmallVector<unsigned> dep;
      for (unsigned idx : phase) {
        Operation *op = ops[idx];
        if (std::find(invariantOps.begin(), invariantOps.end(), op) !=
            invariantOps.end())
          continue;
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
      // Clone the whole phase first, remembering old -> new. Only afterwards
      // are the operands remapped: a clone of this phase may use the result
      // of ANOTHER op of the same phase, and that other op is itself only
      // alive as a clone — the original dies with the old kernel.
      DenseMap<Operation *, Operation *> phaseClone;
      for (unsigned idx : phase) {
        Operation *cloned = bodyBuilder.clone(*ops[idx]);
        phaseClone[ops[idx]] = cloned;
      }
      for (unsigned idx : phase) {
        Operation *cloned = phaseClone[ops[idx]];
        SmallVector<Value> newOperands;
        for (Value operand : cloned->getOperands()) {
          Operation *def = operand.getDefiningOp();
          if (def) {
            if (isa<gpu::ThreadIdOp>(def)) {
              newOperands.push_back(tidValue);
              continue;
            }
            auto inv = invariantClone.find(def);
            if (inv != invariantClone.end()) {
              newOperands.push_back(
                  inv->second->getResult(cast<OpResult>(operand).getResultNumber()));
              continue;
            }
            auto ph = phaseClone.find(def);
            if (ph != phaseClone.end()) {
              newOperands.push_back(
                  ph->second->getResult(cast<OpResult>(operand).getResultNumber()));
              continue;
            }
          }
          newOperands.push_back(operand);
        }
        cloned->setOperands(newOperands);
        if (auto tid = dyn_cast<gpu::ThreadIdOp>(cloned)) {
          if (tid.getDimension() != gpu::Dimension::x) {
            tid.emitError("rouge-simt-to-scf: only the x dimension of "
                          "gpu.thread_id is supported");
            phaseFailed = true;
            return failure();
          }
          tid.getResult().replaceAllUsesWith(tidValue);
          tid->erase();
          continue;
        }
        if (auto bid = dyn_cast<gpu::BlockIdOp>(cloned)) {
          if (bid.getDimension() != gpu::Dimension::x) {
            bid.emitError("rouge-simt-to-scf: only the x dimension of "
                          "gpu.block_id is supported");
            phaseFailed = true;
            return failure();
          }
          if (blockIdArg) {
            bid.getResult().replaceAllUsesWith(blockIdArg);
            bid->erase();
            continue;
          }
        }
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
  LogicalResult convertLaunch(gpu::LaunchFuncOp launch) {
    Location loc = launch.getLoc();
    ModuleOp module = launch->getParentOfType<ModuleOp>();
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
    // body uses gpu.block_id; passing it unconditionally would make the call
    // have an extra operand of the wrong type).
    SmallVector<Value> args(launch.getKernelOperands().begin(),
                            launch.getKernelOperands().end());
    const std::string calleeName = launch.getKernelAttr().getLeafReference().str();
    auto kernelIt = kernelTakesBlockId_.find(calleeName);
    const bool takesBlockId =
        kernelIt != kernelTakesBlockId_.end() && kernelIt->second;
    if (takesBlockId) {
      // The block id argument is appended to the kernel signature, so its
      // type is the one the kernel expects: the same as the thread id.
      Type blockIdType = builder.getIndexType();
      auto kernel = module.lookupSymbol<func::FuncOp>(calleeName);
      if (kernel && !kernel.getFunctionType().getInputs().empty())
        blockIdType = kernel.getFunctionType().getInputs().back();
      Value blockId = forOp.getInductionVar();
      if (blockId.getType() != blockIdType)
        blockId = makeIndexCast(builder, loc, blockIdType, blockId);
      args.push_back(blockId);
    }

    func::CallOp::create(builder, launch.getKernelAttr().getLeafReference(),
                         launch.getResultTypes(), args);
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

}  // namespace

std::unique_ptr<Pass> createRougeSimtToScfPass() {
  return std::make_unique<SimtToScf>();
}

void registerSimtToScfPass() { PassRegistration<SimtToScf>(); }

}  // namespace rouge
