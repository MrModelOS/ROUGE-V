//=- ROUGE-V: SIMT -> SCF (the first rewriting pass of the MLIR contour).
//
// The transformation itself lives in SimtToScfImpl.h — it is shared with the
// pass for OUR dialect (RougeToScf.cpp), because the two dialects describe
// the same thing and a duplicate of the lowering would mean two places to
// fix the same subtle bug.
//
// This file is the driver for the upstream `gpu` dialect: it walks the
// kernels, finds the launch that calls each of them, and calls the shared
// lowering.
//
//=.

#include "rouge/Transforms/Passes.h"

#include "SimtToScfImpl.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include <memory>

using namespace mlir;

namespace rouge {
namespace {

struct SimtToScf
    : public PassWrapper<SimtToScf, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimtToScf)

  using Lowering = SimtLoweringBase<
      gpu::GPUFuncOp, gpu::LaunchFuncOp, gpu::ThreadIdOp, gpu::BlockIdOp,
      gpu::BlockDimOp, gpu::GridDimOp, gpu::BarrierOp, gpu::ReturnOp,
      gpu::Dimension>;

  // The dialects whose ops this pass creates must be LOADED: an input
  // without scf ops never loads the SCF dialect, and creating an op of an
  // unloaded dialect is refused ("created with unregistered dialect").
  // Declaring them as dependencies makes the framework load them BEFORE the
  // pass runs — loading them from inside the pass would race with the
  // framework (and with other passes running in parallel).
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect>();
  }

  StringRef getArgument() const final { return "rouge-simt-to-scf"; }
  StringRef getDescription() const final {
    return "Turn SIMT thread/block ids into explicit loops (phase "
           "decomposition at barriers)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verify(module))) {
      module.emitError("rouge-simt-to-scf: module verification failed");
      return;
    }

    // The launch that calls each kernel (if any): the launch owns the grid
    // and the block sizes, the kernel owns the body.
    SmallVector<gpu::GPUFuncOp> kernels;
    module.walk([&](gpu::GPUFuncOp f) { kernels.push_back(f); });

    unsigned converted = 0, refused = 0;
    for (gpu::GPUFuncOp kernel : kernels) {
      gpu::LaunchFuncOp launch = findLaunch(module, kernel);
      bool takesBlockId = false;
      if (failed(Lowering::convertKernel(kernel, launch, takesBlockId))) {
        ++refused;
        continue;
      }
      if (launch) {
        if (failed(Lowering::convertLaunch(launch, takesBlockId))) {
          ++refused;
          continue;
        }
      }
      ++converted;
    }
    // A gpu.module whose kernels were all converted is now empty and has no
    // business staying in the module: the LLVM lowering of the contour does
    // not know the GPU dialect and would refuse the whole module.
    SmallVector<gpu::GPUModuleOp> gpuModules;
    module.walk([&](gpu::GPUModuleOp m) { gpuModules.push_back(m); });
    unsigned dropped = 0;
    for (gpu::GPUModuleOp m : gpuModules) {
      if (m.getBody()->empty()) {
        m->erase();
        ++dropped;
      }
    }
    llvm::errs() << "rouge-simt-to-scf: kernels converted=" << converted
                 << " refused=" << refused
                 << " empty gpu.modules dropped=" << dropped << "\n";
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
};

}  // namespace

std::unique_ptr<Pass> createRougeSimtToScfPass() {
  return std::make_unique<SimtToScf>();
}

void registerSimtToScfPass() { PassRegistration<SimtToScf>(); }

}  // namespace rouge
