// ROUGE-V contour: input for the SIMT->SCF pass with a DIVERGENT branch.
//
// A branch whose condition depends on the thread id means different threads
// take different paths. The pass must REFUSE such a kernel with a loud
// error: silently serialising it (or pretending the branch is uniform)
// would change the meaning of the kernel. The refusal is the contract, and
// the test checks that the pass reports it instead of producing IR.
//
// NOTE: the file must be valid MLIR — the pass verifies the module before
// rewriting it, so a type error here is a hard failure, not a warning.

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @divergent(%dst: memref<16xf32>) kernel {
      %c0 = arith.constant 0 : index
      %tid = gpu.thread_id x
      %c1 = arith.constant 1 : index
      %cmp = arith.cmpi sgt, %tid, %c1 : index
      %v = memref.load %dst[%c0] : memref<16xf32>
      scf.if %cmp {
        memref.store %v, %dst[%tid] : memref<16xf32>
      } else {
        memref.store %v, %dst[%c0] : memref<16xf32>
      }
      gpu.return
    }
  }
}
