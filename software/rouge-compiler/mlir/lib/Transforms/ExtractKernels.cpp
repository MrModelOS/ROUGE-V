//=- ROUGE-V: rouge-extract-kernels — keep only the device side of the module.
//
//   After the NVPTX lowering chain
//
//     rouge-opt --rouge-to-gpu              (the dialect -> gpu)
//     mlir-opt  --convert-gpu-to-nvvm ...   (gpu -> LLVM dialect, NVVM)
//
//   the modules look like this:
//
//     module attributes {gpu.container_module} {
//       llvm.func @kernel_host(...)      <- the host stub, gpu.launch_func inside
//       gpu.module @kernel_gpu_module {
//         llvm.func @kernel(...) {gpu.kernel, nvvm.kernel}   <- the device code
//       }
//     }
//
//   We only compile the DEVICE half: a .ptx has no host stub, and the host
//   launch is a separate, runtime-specific thing. This pass flattens the
//   gpu.modules into the top-level module and erases everything that is not a
//   device function or a global. The result is a plain LLVM-dialect module
//   that mlir-translate accepts.
//
//   The device functions are not renamed and keep the `nvvm.kernel`
//   attribute, so the NVPTX backend emits them as .entry.
//
//=.

#include "rouge/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace rouge {
namespace {

struct ExtractKernels : public PassWrapper<ExtractKernels,
                                           OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractKernels)

  StringRef getArgument() const final { return "rouge-extract-kernels"; }
  StringRef getDescription() const final {
    return "Flatten the gpu.modules into the top-level module and drop the "
           "host side, leaving a device-only LLVM-dialect module";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The gpu modules: their bodies are already LLVM dialect functions at
    // this point (--convert-gpu-to-nvvm ran), so they can simply be spliced
    // into the parent block.
    SmallVector<gpu::GPUModuleOp> gpuModules;
    module.walk([&](gpu::GPUModuleOp m) { gpuModules.push_back(m); });
    if (gpuModules.empty()) {
      module.emitError("no gpu.module in the module: run --rouge-to-gpu and "
                       "the LLVM lowering before this pass");
      signalPassFailure();
      return;
    }
    for (gpu::GPUModuleOp gm : gpuModules) {
      Block &body = gm.getBodyRegion().front();
      for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
        op.moveBefore(gm);
      }
      gm.erase();
    }
    module->removeAttr("gpu.container_module");

    // Everything else is the host side: the launch stub, the host function,
    // the constants it used. A .ptx has no host code. A device function is
    // either a kernel (nvvm.kernel) or an external declaration it calls.
    for (Operation &op :
         llvm::make_early_inc_range(module.getBody()->getOperations())) {
      if (isa<LLVM::GlobalOp>(op)) continue;
      if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        if (fn.isDeclaration() ||
            fn->hasAttr("nvvm.kernel") || fn->hasAttr("gpu.kernel"))
          continue;
      }
      op.erase();
    }
  }
};

} // namespace

void registerRougeExtractKernelsPass() {
  PassRegistration<ExtractKernels>();
}

std::unique_ptr<mlir::Pass> createRougeExtractKernelsPass() {
  return std::make_unique<ExtractKernels>();
}

} // namespace rouge
