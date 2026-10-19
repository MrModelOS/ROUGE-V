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
// НАПИСАН, НО ПОКА НЕ ВКЛЮЧЁН (см. mlir/README.md): в MLIR 23 проход
// требует доводки, а контур обязан оставаться зелёным.
// std::unique_ptr<mlir::Pass> createRougeSimtToScfPass();

void registerRougeSimtTransformsPasses();


}  // namespace rouge

#endif  // ROUGE_TRANSFORMS_PASSES_H
