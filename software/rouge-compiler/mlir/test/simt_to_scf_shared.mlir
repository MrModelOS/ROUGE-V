// ROUGE-V contour: a kernel that uses SHARED memory — must be REFUSED.
//
// In the GPU dialect the shared memory is a memref with a non-default memory
// space. The new plain function would keep the memref, but the per-block
// copy of the shared memory would be lost, and the accesses would silently
// read the wrong memory. That is a quiet miscompile, so the pass refuses.
//
// NOTE: the file must be valid MLIR — the pass verifies the module before
// rewriting it, so a type error here is a hard failure, not a warning.

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @with_shared(%dst: memref<16xf32, 3>) kernel {
      %c0 = arith.constant 0 : index
      %tid = gpu.thread_id x
      %v = memref.load %dst[%tid] : memref<16xf32, 3>
      memref.store %v, %dst[%tid] : memref<16xf32, 3>
      gpu.return
    }
  }
}
