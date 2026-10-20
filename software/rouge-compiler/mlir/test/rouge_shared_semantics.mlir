// ROUGE-V: the SEMANTIC check of the shared-memory lowering.
//
// The kernel uses the shared memory the way CUDA does: the declaration is
// INSIDE the kernel (a memref.alloc with the shared address space), the
// threads fill it, the barrier separates the phases, and the threads read
// the values back.
//
// The lowering turns the declaration into a private allocation: the kernel
// runs once per block, so the allocation is per block — exactly the lifetime
// the shared memory has. The barrier keeps the phases apart, so the ordering
// requirement holds.
//
// The result must be the same as the reference: every element of dst is the
// corresponding element of src, so the sum is 64 * 1.5 = 96.

module {
  func.func @shared_copy(%src: memref<64xf32>, %dst: memref<64xf32>) {
    %shared = memref.alloc() : memref<64xf32, 3>
    %tid = rouge.thread_id x : index
    %v = memref.load %src[%tid] : memref<64xf32>
    memref.store %v, %shared[%tid] : memref<64xf32, 3>
    rouge.barrier
    %w = memref.load %shared[%tid] : memref<64xf32, 3>
    memref.store %w, %dst[%tid] : memref<64xf32>
    memref.dealloc %shared : memref<64xf32, 3>
    return
  }

  func.func @main() {
    %src = memref.alloc() : memref<64xf32>
    %dst = memref.alloc() : memref<64xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %f = arith.constant 1.500000e+00 : f32
    scf.for %i = %c0 to %c64 step %c1 {
      memref.store %f, %src[%i] : memref<64xf32>
    }
    %grid = arith.constant 1 : index
    %block = arith.constant 64 : index
    "rouge.launch"(%grid, %grid, %grid,
                    %block, %block, %block,
                    %src, %dst) {kernel = @shared_copy}
      : (index, index, index, index, index, index,
         memref<64xf32>, memref<64xf32>) -> ()
    %zero = arith.constant 0.000000e+00 : f32
    %sum = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %zero) -> f32 {
      %v = memref.load %dst[%i] : memref<64xf32>
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    }
    func.call @print_f32(%sum) : (f32) -> ()
    memref.dealloc %src : memref<64xf32>
    memref.dealloc %dst : memref<64xf32>
    return
  }

  func.func private @print_f32(f32)
}
