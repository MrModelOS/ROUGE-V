// ROUGE-V contour: input for the unit-stride -> vector pass.
//
// The kernel copies 64 f32 from src to dst. The pass must split the thread
// loop into a vector part (4 elements per iteration) and an exact scalar
// tail.
//
// The module also carries a main() that fills src with a known pattern, runs
// the copy and prints the sum of dst. The SEMANTIC check runs this module
// twice — once as written, once after the pass — and compares the printed
// sums: the vectorised kernel must compute exactly the same thing.
//
// NOTE: the file must be valid MLIR — the pass verifies the module before
// rewriting it, so a type error here is a hard failure, not a warning.

func.func @copy(%src: memref<64xf32>, %dst: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c64 step %c1 {
    %v = memref.load %src[%i] : memref<64xf32>
    memref.store %v, %dst[%i] : memref<64xf32>
  }
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
  func.call @copy(%src, %dst) : (memref<64xf32>, memref<64xf32>) -> ()
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
