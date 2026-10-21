// ROUGE-V: the divergence model — a mask instead of the serialisation.
//
// The kernel's thread loop has a DIVERGENT branch: the threads with the
// index below 32 take it, the others do not. In the scalar form that is an
// scf.if inside the loop (the threads are serialised: each one takes its own
// path — correct, but the vector unit idles).
//
// In the vector form the branch becomes a MASK: the condition is computed
// per lane (the lane-wise thread ids against the constant), and the store
// inside the region becomes a masked store — the lanes that took the other
// path write nothing.
//
// The result must be the same as the scalar one: the first half of dst is
// copied, the second half is untouched.

module {
  func.func @divergent(%src: memref<64xf32>, %dst: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c64 step %c1 {
      %v = memref.load %src[%i] : memref<64xf32>
      %cond = arith.cmpi slt, %i, %c32 : index
      scf.if %cond {
        memref.store %v, %dst[%i] : memref<64xf32>
      }
    }
    return
  }

  func.func @main() {
    %src = memref.alloc() : memref<64xf32>
    %dst = memref.alloc() : memref<64xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %f = arith.constant 1.500000e+00 : f32
    %zero = arith.constant 0.000000e+00 : f32
    scf.for %i = %c0 to %c64 step %c1 {
      memref.store %f, %src[%i] : memref<64xf32>
      memref.store %zero, %dst[%i] : memref<64xf32>
    }
    func.call @divergent(%src, %dst) : (memref<64xf32>, memref<64xf32>) -> ()
    // The sum of the first half and of the second half: the first is 32
    // elements of 1.5, the second is untouched (zeros).
    %sum1 = scf.for %i = %c0 to %c32 step %c1 iter_args(%acc = %zero) -> f32 {
      %v = memref.load %dst[%i] : memref<64xf32>
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    }
    %sum2 = scf.for %i = %c32 to %c64 step %c1 iter_args(%acc = %zero) -> f32 {
      %v = memref.load %dst[%i] : memref<64xf32>
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    }
    func.call @print_f32(%sum1) : (f32) -> ()
    func.call @print_f32(%sum2) : (f32) -> ()
    memref.dealloc %src : memref<64xf32>
    memref.dealloc %dst : memref<64xf32>
    return
  }

  func.func private @print_f32(f32)
}
