//===- RougeOps.td - ROUGE-V dialect definition -*- tablegen -*-===//
//
// The ROUGE dialect: our own representation of a SIMT kernel, on the way from
// PTX to a vector form for a CPU.
//
// Why our own dialect and not the upstream `gpu` one:
//
//   * the upstream GPU dialect models the GPU's execution model, and our
//     target is a CPU: a thread is an iteration of a loop, not a hardware
//     lane. The `gpu` dialect's block/thread ids and its barrier carry
//     hardware semantics we would have to fight at every step;
//   * the ADDRESS SPACE has to be part of the type. The memory accesses stay
//     in the upstream `memref` dialect, whose type carries the memory space:
//     `memref<64xf32>` is the global memory, `memref<64xf32, 3>` is the
//     shared one. So "shared" and "global" cannot be mixed up by
//     construction, and the CPU lowering can turn shared into private
//     storage by the type alone;
//   * a kernel that uses shared memory, or a barrier inside a divergence, has
//     to be REFUSED loudly. Writing that policy in our own dialect is
//     straightforward; in someone else's dialect it is a fight with their
//     verification rules.
//
// The arithmetic itself is NOT reinvented: element-wise ops come from the
// upstream `arith` dialect (they are element-wise, so they vectorise for
// free). What we define is what is specific to a SIMT kernel: the thread and
// block identity, the sizes, the barrier and the memory access with an
// explicit address space.
//
//===----------------------------------------------------------------------===//

#ifndef ROUGE_OPS_TD
#define ROUGE_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Rouge_Dialect : Dialect {
  let name = "rouge";
  let cppNamespace = "::rouge";
  let summary = "The ROUGE dialect: SIMT kernels on the way to vector code";
  let description = [{
    The ROUGE dialect describes a SIMT kernel: the identity and the sizes of
    the threads and the blocks, the barrier between the phases of a kernel and
    the memory accesses with an explicit address space.

    The dialect is deliberately small. Everything that is element-wise
    (arithmetic, comparisons, conversions) stays in the upstream `arith`
    dialect: re-implementing it would mean re-implementing the vectoriser
    with it.

    Example:

    ```mlir
    func.func @copy(%src: memref<64xf32>, %dst: memref<64xf32>) {
      %tid = rouge.thread_id x : index
      %v = rouge.load %src[%tid] : memref<64xf32>
      rouge.store %v, %dst[%tid] : memref<64xf32>
      rouge.barrier
      return
    }
    ```
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

// An enum in the `rouge` C++ namespace (the upstream I32Enum leaves the
// namespace to the def, and a class body cannot be opened on a def).
class Rouge_I32Enum<string name, string description, list<EnumCase> cases>
    : I32Enum<name, description, cases> {
  let cppNamespace = "::rouge";
}

def Rouge_Dimension : Rouge_I32Enum<"Dimension", "a dimension: x, y or z", [
    EnumCase<"x", 0, "x", 32>,
    EnumCase<"y", 1, "y", 32>,
    EnumCase<"z", 2, "z", 32>
  ]>;

// The dimension prints as a bare keyword (x, y, z) — the same way the
// upstream gpu dialect prints it.
def Rouge_DimensionAttr : EnumAttr<Rouge_Dialect, Rouge_Dimension, "dim">;

class Rouge_Op<string mnemonic, list<Trait> traits = []>
    : Op<Rouge_Dialect, mnemonic, traits>;

class Rouge_IndexOp<string mnemonic>
    : Rouge_Op<mnemonic, [Pure]> {
  let arguments = (ins Rouge_DimensionAttr:$dimension);
  let results = (outs Index:$result);
  let assemblyFormat = "$dimension attr-dict `:` type($result)";
}

def Rouge_ThreadIdOp : Rouge_IndexOp<"thread_id"> {
  let summary = "the index of the thread inside its block";
  let description = [{
    Returns the index of the calling thread within its block, along the given
    dimension. In the lowered (loop) form this is the induction variable of
    the thread loop.
  }];
}

def Rouge_BlockIdOp : Rouge_IndexOp<"block_id"> {
  let summary = "the index of the block inside the grid";
}

def Rouge_BlockDimOp : Rouge_IndexOp<"block_dim"> {
  let summary = "the number of threads in a block, along the dimension";
}

def Rouge_GridDimOp : Rouge_IndexOp<"grid_dim"> {
  let summary = "the number of blocks in the grid, along the dimension";
}

// The barrier's memory effects are not declared on purpose: an op with
// UNKNOWN effects is the safe direction for every analysis (the barrier
// synchronises memory, and saying "writes" would be a lie about the reads).
def Rouge_BarrierOp : Rouge_Op<"barrier"> {
  let summary = "all the threads of the block reach this point";
  let description = [{
    The barrier of the SIMT execution model: once one thread of a block has
    executed this barrier, every other thread of the same block must execute
    the same dynamic instance of it before any thread executes any other
    instance. A barrier inside a divergent branch breaks that contract, and
    the lowering refuses such kernels loudly.
  }];
  let assemblyFormat = "attr-dict";
}

def Rouge_LaunchOp : Rouge_Op<"launch"> {
  let summary = "launch a kernel over a grid of blocks";
  let description = [{
    Runs a kernel over a grid of blocks. The host side of the execution: the
    kernel is a `func.func` (after the outlining) and the grid and block
    sizes are the loop bounds of the launch.
  }];
  let arguments = (ins SymbolRefAttr:$kernel,
                       Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,
                       Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
                       Variadic<AnyType>:$kernelOperands);
  // The launch is printed in the GENERIC form: a variadic of any type has
  // no printable declarative form, and a custom printer would be more code
  // than the op deserves (it is an internal op of the contour).
  let hasCustomAssemblyFormat = 0;
}

#endif // ROUGE_OPS_TD
