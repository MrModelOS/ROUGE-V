// ROUGE-V: a rouge-dialect kernel that uses SHARED memory.
//
// The shared memory is a memref with a non-default memory space
// (memref<64xf32, 3>): the ADDRESS SPACE is part of the type, so the
// lowering cannot mistake it for the global memory.
//
// The pass must refuse the kernel loudly: the per-block copy of the shared
// memory would be lost silently, and the accesses would read the wrong
// memory. The refusal is the honest answer until the shared memory becomes
// a per-block private storage.

module {
  func.func @shared_copy(%src: memref<64xf32, 3>, %dst: memref<64xf32>) {
    %tid = rouge.thread_id x : index
    %v = memref.load %src[%tid] : memref<64xf32, 3>
    memref.store %v, %dst[%tid] : memref<64xf32>
    return
  }

  func.func @host() {
    %grid_x = arith.constant 1 : index
    %block_x = arith.constant 64 : index
    %src = memref.alloc() : memref<64xf32, 3>
    %dst = memref.alloc() : memref<64xf32>
    "rouge.launch"(%grid_x, %grid_x, %grid_x,
                    %block_x, %block_x, %block_x,
                    %src, %dst) {kernel = @shared_copy}
      : (index, index, index, index, index, index,
         memref<64xf32, 3>, memref<64xf32>) -> ()
    memref.dealloc %src : memref<64xf32, 3>
    memref.dealloc %dst : memref<64xf32>
    return
  }
}
