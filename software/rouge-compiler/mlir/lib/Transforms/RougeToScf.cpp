//=- ROUGE-V: rouge-to-scf — the SIMT -> SCF lowering for OUR dialect.
//
// The transformation itself lives in SimtToScfImpl.h — it is shared with the
// pass for the upstream `gpu` dialect (SimtToScf.cpp), because the two
// dialects describe the same thing and a duplicate of the lowering would
// mean two places to fix the same subtle bug.
//
// The difference is in the driver: in our dialect the kernel is a plain
// `func.func` and the launch is a `rouge.launch` op, so the pass walks the
// LAUNCHES (a plain function is not necessarily a kernel) and finds the
// kernel by its symbol.
//
//   rouge.launch          -> a loop over the grid
//   the kernel body       -> PHASES at the barriers, one thread loop each
//   rouge.thread_id x     -> the induction variable of the thread loop
//   rouge.block_id x      -> an argument of the kernel function
//   rouge.block_dim x     -> the bound of the thread loop
//   rouge.grid_dim x      -> the bound of the block loop
//   rouge.load / .store   -> memref.load / memref.store (the address space
//                             stays in the memref type)
//
//   rouge-opt --rouge-to-scf kernel.mlir
//
//=.

#include "rouge/Transforms/Passes.h"

#include "rouge/IR/RougeDialect.h"

#include "SimtToScfImpl.h"

#include "mlir/IR/BuiltinOps.h"

#include <memory>

using namespace mlir;

namespace rouge {
namespace {

struct RougeToScf
    : public PassWrapper<RougeToScf, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RougeToScf)

  using Lowering = SimtLoweringBase<
      func::FuncOp, rouge::LaunchOp, rouge::ThreadIdOp, rouge::BlockIdOp,
      rouge::BlockDimOp, rouge::GridDimOp, rouge::BarrierOp, func::ReturnOp,
      rouge::Dimension>;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    rouge::RougeDialect>();
  }

  StringRef getArgument() const final { return "rouge-to-scf"; }
  StringRef getDescription() const final {
    return "Lower a rouge-dialect SIMT kernel to loops (phase decomposition "
           "at barriers)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verify(module))) {
      module.emitError("rouge-to-scf: module verification failed");
      return;
    }

    SmallVector<rouge::LaunchOp> launches;
    module.walk([&](rouge::LaunchOp l) { launches.push_back(l); });

    unsigned converted = 0, refused = 0;
    for (rouge::LaunchOp launch : launches) {
      func::FuncOp kernel =
          Lowering::findKernel(module, launch);
      if (!kernel) {
        ++refused;
        continue;
      }
      bool takesBlockId = false;
      if (failed(Lowering::convertKernel(kernel, launch, takesBlockId))) {
        ++refused;
        continue;
      }
      if (failed(Lowering::convertLaunch(launch, takesBlockId))) {
        ++refused;
        continue;
      }
      ++converted;
    }
    llvm::errs() << "rouge-to-scf: launches converted=" << converted
                 << " refused=" << refused << "\n";
  }
};

}  // namespace

std::unique_ptr<Pass> createRougeToScfPass() {
  return std::make_unique<RougeToScf>();
}

void registerRougeToScfPass() { PassRegistration<RougeToScf>(); }

}  // namespace rouge
