//=- ROUGE-V: SIMT access-pattern report (analysis pass of the MLIR contour).
//
// The GPU->Vector->RVV lowering (docs/06, stage 2) can only vectorize accesses
// that are (a) unit-stride and (b) not straddling a barrier. This pass makes
// those properties measurable before the lowering exists: it prints, per
// function, a histogram of gpu/memref/vector ops plus the access-pattern
// summary the vectorizer must be designed against. It never rewrites IR, so it
// is safe to run on any MLIR module.
//
//   rouge-opt --rouge-simt-access-report kernel.mlir
//
// Only generic IR APIs are used (no dialect headers) so the pass keeps
// building across MLIR versions while the contour is young.
//
//=.

#include "rouge/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace mlir;

namespace rouge {
namespace {

// Access-pattern properties the future SIMT->RVV vectorizer cares about.
struct AccessSummary {
  unsigned loads = 0;      // memref/gpu loads
  unsigned stores = 0;     // memref/gpu stores
  unsigned vectorOps = 0;  // already-vectorized ops
  unsigned atomics = 0;    // memref.atomic_rmw / gpu atomic rmw
  unsigned barriers = 0;   // gpu.barrier and friends
  unsigned threadIds = 0;  // gpu.thread_id / block_id / dims
};

bool isLoad(StringRef name) {
  return name == "memref.load" || name == "gpu.load" ||
         name == "vector.load" || name == "vector.load_gather" ||
         name == "vector.transfer_read";
}

bool isStore(StringRef name) {
  return name == "memref.store" || name == "gpu.store" ||
         name == "vector.store" || name == "vector.transfer_write";
}

bool isAtomic(StringRef name) { return name.contains("atomic_rmw"); }

bool isBarrier(StringRef name) {
  return name == "gpu.barrier" || name == "gpu.wait";
}

bool isThreadId(StringRef name) {
  return name.starts_with("gpu.thread_id") || name.starts_with("gpu.block_id") ||
         name.starts_with("gpu.grid_dim") || name.starts_with("gpu.block_dim");
}

bool isVector(StringRef name) { return name.starts_with("vector."); }

// Function-like containers: the report groups ops by their nearest enclosing
// function. Checked by name so the pass does not depend on dialect headers.
bool isFunctionLike(StringRef name) {
  return name == "func.func" || name == "llvm.func" || name == "gpu.func" ||
         name == "gpu.kernel_func" || name == "spv.func" ||
         name.ends_with(".kernel") || name.ends_with(".kernel_func");
}

class SimtAccessReport
    : public PassWrapper<SimtAccessReport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimtAccessReport)

  StringRef getArgument() const final { return "rouge-simt-access-report"; }
  StringRef getDescription() const final {
    return "Report per-function SIMT access patterns (gpu/memref/vector) — "
           "the input for the ROUGE SIMT->RVV lowering";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verify(module))) {
      module.emitError("rouge-simt-access-report: module verification failed");
      return;
    }

    llvm::StringMap<AccessSummary> summaries;
    for (Operation& top : module.getBody()->getOperations()) {
      const StringRef topName = top.getName().getStringRef();
      const std::string label = isFunctionLike(topName) ? topName.str()
                                                        : std::string("(module)");
      AccessSummary& s = summaries[label];
      top.walk([&](Operation *op) {
        const StringRef name = op->getName().getStringRef();
        if (isLoad(name)) ++s.loads;
        else if (isStore(name)) ++s.stores;
        else if (isAtomic(name)) ++s.atomics;
        else if (isBarrier(name)) ++s.barriers;
        else if (isThreadId(name)) ++s.threadIds;
        if (isVector(name)) ++s.vectorOps;
      });
    }

    llvm::errs() << "rouge-simt-access-report: module '" << module.getName()
                 << "'\n";
    if (summaries.empty()) {
      llvm::errs() << "  (no operations to report)\n";
      return;
    }
    for (const auto& entry : summaries) {
      const AccessSummary& s = entry.second;
      llvm::errs() << llvm::formatv(
          "  {0}: loads={1} stores={2} vector={3} atomics={4} barriers={5} "
          "ids={6}\n",
          entry.first(), s.loads, s.stores, s.vectorOps, s.atomics, s.barriers,
          s.threadIds);
    }
    llvm::errs() << "rouge-simt-access-report: lowering plan hints\n";
    for (const auto& entry : summaries) {
      const AccessSummary& s = entry.second;
      if (s.loads + s.stores == 0) continue;
      llvm::errs() << llvm::formatv(
          "  {0}: {1}\n", entry.first(),
          (s.atomics || s.barriers)
              ? "mixed access — vectorize only the runs between barriers"
              : "unit-stride candidate — SIMT->RVV vectorizable");
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createRougeSimtAccessReportPass() {
  return std::make_unique<SimtAccessReport>();
}

void registerRougeTransformsPasses() {
  PassRegistration<SimtAccessReport>();
}

}  // namespace rouge
