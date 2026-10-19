// ROUGE-V contour: input for the SIMT->SCF pass.
//
// The kernel has the shape a real CUDA kernel has: a launch (grid/block
// sizes), the thread id indexing a global array (unit stride) and a barrier
// splitting the body into two phases. After --rouge-simt-to-scf the module
// must contain: the launch turned into a loop over the grid, and inside the
// kernel — one thread loop per phase, with the barrier gone and the thread
// id replaced by the induction variable.
//
// NOTE: the file must be valid MLIR — the pass verifies the module before
// rewriting it, so a type error here is a hard failure, not a warning.

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @scale_copy(%src: memref<1024xf32>, %dst: memref<1024xf32>) kernel {
      %c0 = arith.constant 0 : index
      %tid = gpu.thread_id x
      %v = memref.load %src[%tid] : memref<1024xf32>
      memref.store %v, %dst[%tid] : memref<1024xf32>
      gpu.barrier
      %w = memref.load %src[%c0] : memref<1024xf32>
      memref.store %w, %dst[%c0] : memref<1024xf32>
      gpu.return
    }
  }

  func.func @host() {
    %grid_x = arith.constant 4 : index
    %block_x = arith.constant 256 : index
    %src = memref.alloc() : memref<1024xf32>
    %dst = memref.alloc() : memref<1024xf32>
    gpu.launch_func @kernels::@scale_copy
        blocks in (%grid_x, %grid_x, %grid_x)
        threads in (%block_x, %block_x, %block_x)
        args(%src : memref<1024xf32>, %dst : memref<1024xf32>)
    memref.dealloc %src : memref<1024xf32>
    memref.dealloc %dst : memref<1024xf32>
    return
  }
}
