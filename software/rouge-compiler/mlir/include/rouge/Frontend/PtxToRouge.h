//===- PtxToRouge.h - the PTX front-end of the rouge dialect -*- C++ -*-===//
//
// The front-end: PTX (the text nvcc produces) -> a module of the `rouge`
// dialect.
//
// Why a front-end at all: the dialect is the source of truth of the 2.0
// plan, and a dialect nobody can feed is a toy. The front-end is also
// where the model of memory is decided.
//
// The model: PTX addresses are FLAT (a u64 parameter points into memory,
// and the kernel does the arithmetic itself), while MLIR's memrefs are
// STRUCTURED (a base, a shape, a stride). The front-end therefore recovers
// the structure from the address arithmetic: it recognises the pattern
//
//     cvta.to.global.u64 %base, %param;
//     mul.wide.u32 %off, %index, <elem size>;
//     add.s64 %addr, %base, %off;
//     ld.global.s32 %v, [%addr];
//
// and emits
//
//     %v = memref.load %param[%index] : memref<?xi32>
//
// The recovery is deliberately narrow: only a base that is a kernel
// parameter, an index that is scaled by exactly the element size, and no
// other arithmetic in between. Everything else is REFUSED with a loud
// message — a wrong guess here would read the wrong memory, and a silent
// guess is exactly what this project does not ship.
//
// The kernel's parameters become memrefs. Their SHAPE is not in the PTX
// (the GPU does not care), so it is dynamic by default and can be given as
// a hint (see the tool's options) — the vectoriser needs a static shape,
// and only the caller knows it.
//
//===----------------------------------------------------------------------===//

#ifndef ROUGE_FRONTEND_PTXTOROUGE_H
#define ROUGE_FRONTEND_PTXTOROUGE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include <map>
#include <string>
#include <vector>

namespace rouge {

// The shapes of the kernel's parameters, given by the caller: a map from
// the parameter's index to its shape ("64xf32") and element type ("f32").
// A parameter without an entry gets a dynamic shape.
struct PtxShapeHints {
  std::map<int, std::string> shapes;   // the parameter index -> "64xf32"
  std::map<int, std::string> elemTypes;  // the parameter index -> "f32"
  // The launch configuration: the grid and the block, in threads. The PTX
  // does not carry it (it is a property of the launch, not of the kernel),
  // and only the caller knows it. With block == 0 no launch is emitted: the
  // module then has just the kernel.
  int grid = 1;
  int block = 0;
};

// Translate a parsed PTX program into a module of the `rouge` dialect.
//
// The kernel's body is translated into a plain func.func: the launch
// (rouge.launch) and the identity (rouge.thread_id, rouge.block_id,
// rouge.block_dim, rouge.grid_dim) are what the kernel's body talks about,
// and the lowering turns them into loops.
//
// When the hints carry a block size, a host function with a rouge.launch is
// emitted as well: the launch configuration belongs to the caller, and the
// lowering turns the launch into a loop over the grid.
//
// Returns failure and leaves a diagnostic in `error` when the program uses
// something this front-end does not support — loudly, because a kernel that
// is translated into something else is worse than a kernel that is refused.
mlir::LogicalResult translatePtxToRouge(mlir::ModuleOp module,
                                        const struct PtxProgram &program,
                                        const PtxShapeHints &hints,
                                        std::string *error);

}  // namespace rouge

#endif // ROUGE_FRONTEND_PTXTOROUGE_H
