// ROUGE-V MLIR transforms — pass declarations.
//
// Contour stage (see mlir/README.md and docs/06-compiler-architecture.md):
// the pass here is the *analysis* step of the GPU->Vector->RVV pipeline. It
// reports what a CUDA/PTX kernel actually does in SIMT terms (thread/block
// ids, barriers, global/shared traffic, atomics, vector ops), which is the
// input the SIMT->RVV lowering must be designed against — vectorize only what
// the report says is unit-stride and barrier-free.

#ifndef ROUGE_TRANSFORMS_PASSES_H
#define ROUGE_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace rouge {

// !rouge-simt-access-report — walk the module and print, per function, a
// histogram of the gpu/memref/vector operations plus the access-pattern
// summary used to plan the SIMT->RVV lowering. Read-only pass: it never
// rewrites IR.
std::unique_ptr<mlir::Pass> createRougeSimtAccessReportPass();

// !rouge-simt-to-scf — the first rewriting pass: the SIMT thread/block ids
// become explicit loops (scf.for), the body is split into phases at barriers
// so that the barrier's ordering requirement survives the transformation.
std::unique_ptr<mlir::Pass> createRougeSimtToScfPass();

void registerRougeSimtTransformsPasses();

// Registers the rewriting pass (its class lives in an anonymous namespace in
// another translation unit).
void registerSimtToScfPass();

// !rouge-unit-stride-to-vector — vectorises the unit-stride accesses of a
// thread loop: the loop is split into a vector part (V elements per
// iteration) and an exact scalar tail.
std::unique_ptr<mlir::Pass> createRougeUnitStrideToVectorPass();
void registerRougeUnitStrideToVectorPass();

// !rouge-to-scf — the SIMT -> SCF lowering for OUR dialect: the launch
// becomes a loop over the grid, the body is split into phases at the
// barriers, the thread id becomes the induction variable of the thread
// loop, and the address space stays in the memref type.
std::unique_ptr<mlir::Pass> createRougeToScfPass();
void registerRougeToScfPass();


}  // namespace rouge

#endif  // ROUGE_TRANSFORMS_PASSES_H
