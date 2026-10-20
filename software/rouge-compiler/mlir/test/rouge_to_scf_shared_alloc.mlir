// ROUGE-V: a rouge-dialect kernel with SHARED memory declared inside it.
//
// The shared memory is a memref with a non-default memory space
// (memref<64xf32, 3>): the ADDRESS SPACE is part of the type, so the
// lowering cannot mistake it for the global memory.
//
// The lowering must ACCEPT this kernel: the declaration becomes a private
// allocation, and since the kernel runs once per block, the allocation is
// per block — exactly the lifetime the shared memory has. The barrier splits
// the body into phases, so the ordering requirement of the shared memory
// holds.

module {
  func.func @shared_kernel(%dst: memref<64xf32>) {
    %shared = memref.alloc() : memref<64xf32, 3>
    %tid = rouge.thread_id x : index
    %v = memref.load %shared[%tid] : memref<64xf32, 3>
    memref.store %v, %dst[%tid] : memref<64xf32>
    rouge.barrier
    %w = memref.load %shared[%tid] : memref<64xf32, 3>
    memref.store %w, %dst[%tid] : memref<64xf32>
    memref.dealloc %shared : memref<64xf32, 3>
    return
  }

  func.func @host() {
    %grid_x = arith.constant 1 : index
    %block_x = arith.constant 64 : index
    %dst = memref.alloc() : memref<64xf32>
    "rouge.launch"(%grid_x, %grid_x, %grid_x,
                    %block_x, %block_x, %block_x,
                    %dst) {kernel = @shared_kernel}
      : (index, index, index, index, index, index, memref<64xf32>) -> ()
    memref.dealloc %dst : memref<64xf32>
    return
  }
}
