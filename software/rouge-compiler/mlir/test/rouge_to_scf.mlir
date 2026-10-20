// ROUGE-V: input for the rouge-to-scf pass (OUR dialect).
//
// The same kernel as in simt_to_scf.mlir, but written in the `rouge`
// dialect instead of the upstream `gpu` one:
//
//   * the kernel is a plain func.func (not a gpu.func in a gpu.module);
//   * the launch is a rouge.launch op (not a gpu.launch_func);
//   * the memory accesses are the upstream memref.load / memref.store, and
//     the ADDRESS SPACE stays in the memref type (memref<1024xf32, 3> would
//     be the shared memory).
//
// After --rouge-to-scf the module must contain: the launch turned into a
// loop over the grid, and inside the kernel — one thread loop per phase
// (the barrier splits the body into two phases), with the barrier gone and
// the thread id replaced by the induction variable.
//
// NOTE: the file must be valid MLIR — the pass verifies the module before
// rewriting it, so a type error here is a hard failure, not a warning.

module {
  func.func @scale_copy(%src: memref<1024xf32>, %dst: memref<1024xf32>) {
    %tid = rouge.thread_id x : index
    %v = memref.load %src[%tid] : memref<1024xf32>
    memref.store %v, %dst[%tid] : memref<1024xf32>
    rouge.barrier
    %c0 = arith.constant 0 : index
    %w = memref.load %src[%c0] : memref<1024xf32>
    memref.store %w, %dst[%c0] : memref<1024xf32>
    return
  }

  func.func @host() {
    %grid_x = arith.constant 4 : index
    %block_x = arith.constant 256 : index
    %src = memref.alloc() : memref<1024xf32>
    %dst = memref.alloc() : memref<1024xf32>
    "rouge.launch"(%grid_x, %grid_x, %grid_x,
                    %block_x, %block_x, %block_x,
                    %src, %dst) {kernel = @scale_copy}
      : (index, index, index, index, index, index,
         memref<1024xf32>, memref<1024xf32>) -> ()
    memref.dealloc %src : memref<1024xf32>
    memref.dealloc %dst : memref<1024xf32>
    return
  }
}
