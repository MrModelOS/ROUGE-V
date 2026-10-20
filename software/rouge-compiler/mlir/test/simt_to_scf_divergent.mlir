// ROUGE-V contour: a kernel with a DIVERGENT branch (no barrier inside it).
//
// A branch whose condition depends on the thread id is NOT an error: our
// lowering runs the threads sequentially, so each thread simply takes its own
// path — the same semantics the hardware has, minus the parallelism. Without
// a barrier the order between threads is undefined on a GPU anyway, so a
// sequential order is one valid instance of it.
//
// The pass must therefore LOWER this kernel (not refuse it), and the result
// must still verify.
//
// NOTE: the file must be valid MLIR — the pass verifies the module before
// rewriting it, so a type error here is a hard failure, not a warning.

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @divergent(%dst: memref<16xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
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
