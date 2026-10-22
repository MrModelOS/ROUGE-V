//=- ROUGE-V: rouge-to-gpu — the reverse lowering of our dialect to the
//   upstream `gpu` dialect, the first step towards an NVPTX target.
//=.

#include "rouge/Transforms/Passes.h"

#include "rouge/IR/RougeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace rouge {
namespace {

struct RougeToGpu : public PassWrapper<RougeToGpu, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RougeToGpu)

  StringRef getArgument() const final { return "rouge-to-gpu"; }
  StringRef getDescription() const final {
    return "Lower the rouge dialect to the upstream gpu dialect (step towards "
           "an NVPTX target)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.getContext()->loadDialect<gpu::GPUDialect>();
    SmallVector<Operation *> toConvert;
    module.walk([&](Operation *op) {
      if (isa<rouge::ThreadIdOp, rouge::BlockIdOp, rouge::BlockDimOp,
              rouge::GridDimOp, rouge::BarrierOp>(op))
        toConvert.push_back(op);
    });
    for (Operation *op : toConvert) {
      if (auto tid = dyn_cast<rouge::ThreadIdOp>(op)) {
        OpBuilder b(tid.getContext());
        b.setInsertionPoint(tid);
        auto gpuOp = b.create<gpu::ThreadIdOp>(tid.getLoc(),
                                               b.getIndexType(),
                                               gpu::Dimension::x);
        tid.getResult().replaceAllUsesWith(gpuOp.getResult());
        tid.erase();
      } else if (auto bid = dyn_cast<rouge::BlockIdOp>(op)) {
        OpBuilder b(bid.getContext());
        b.setInsertionPoint(bid);
        auto gpuOp = gpu::BlockIdOp::create(b, bid.getLoc(),
                                            static_cast<gpu::Dimension>(bid.getDimension()));
        bid.getResult().replaceAllUsesWith(gpuOp.getResult());
        bid.erase();
      } else if (auto bdim = dyn_cast<rouge::BlockDimOp>(op)) {
        OpBuilder b(bdim.getContext());
        b.setInsertionPoint(bdim);
        auto gpuOp = gpu::BlockDimOp::create(b, bdim.getLoc(),
                                            static_cast<gpu::Dimension>(bdim.getDimension()));
        bdim.getResult().replaceAllUsesWith(gpuOp.getResult());
        bdim.erase();
      } else if (auto gdim = dyn_cast<rouge::GridDimOp>(op)) {
        OpBuilder b(gdim.getContext());
        b.setInsertionPoint(gdim);
        auto gpuOp = gpu::GridDimOp::create(b, gdim.getLoc(),
                                            static_cast<gpu::Dimension>(gdim.getDimension()));
        gdim.getResult().replaceAllUsesWith(gpuOp.getResult());
        gdim.erase();
      } else if (auto bar = dyn_cast<rouge::BarrierOp>(op)) {
        OpBuilder b(bar.getContext());
        b.setInsertionPoint(bar);
        b.create<gpu::BarrierOp>(bar.getLoc());
        bar.erase();
      }
    }
  }
};

} // namespace

void registerRougeToGpuPass() {
  PassRegistration<RougeToGpu>();
}

std::unique_ptr<mlir::Pass> createRougeToGpuPass() {
  return std::make_unique<RougeToGpu>();
}

} // namespace rouge