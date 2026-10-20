// ROUGE-V contour: the FULL chain input for OUR dialect — rouge -> SCF ->
// Vector -> RVV.
//
// The same kernel as in simt_to_rvv.mlir, but written in the `rouge`
// dialect: the launch is a rouge.launch, the kernel is a plain func.func and
// the thread id is a rouge.thread_id. The chain is:
//   rouge.thread_id -> scf.for -> vector loop + scalar tail -> RVV
// instructions.
//
// The module also carries a main() that fills the source, runs the copy and
// prints the sum: the scalar and the fully lowered builds must print the same
// number, and the RISC-V build must contain real vector instructions.
//
// This test is the strongest evidence that our dialect is a REPLACEMENT of
// the upstream gpu dialect and not a rename: the same kernel, written in the
// two dialects, goes through the same chain and computes the same thing.
//
// NOTE: the file must be valid MLIR — the passes verify the module before
// rewriting it, so a type error here is a hard failure, not a warning.

module {
  func.func @copy(%src: memref<64xf32>, %dst: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %tid = rouge.thread_id x : index
    %v = memref.load %src[%tid] : memref<64xf32>
    memref.store %v, %dst[%tid] : memref<64xf32>
    rouge.barrier
    %w = memref.load %src[%c0] : memref<64xf32>
    memref.store %w, %dst[%c0] : memref<64xf32>
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
                    %src, %dst) {kernel = @copy}
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
