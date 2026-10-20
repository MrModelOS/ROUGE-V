// ROUGE-V contour: the FULL chain input — SIMT (GPU dialect) -> SCF -> Vector
// -> RVV.
//
// The kernel is a plain element-wise copy over the thread index, with a
// barrier in the middle (so the SCF pass has to split the body into phases).
// The chain is:
//   gpu.thread_id -> scf.for -> vector loop + scalar tail -> RVV instructions.
//
// The module also carries a main() that fills the source, runs the copy and
// prints the sum: the scalar and the fully lowered builds must print the same
// number, and the RISC-V build must contain real vector instructions.
//
// NOTE: the file must be valid MLIR — the passes verify the module before
// rewriting it, so a type error here is a hard failure, not a warning.

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @copy(%src: memref<64xf32>, %dst: memref<64xf32>) kernel {
      %c0 = arith.constant 0 : index
      %tid = gpu.thread_id x
      %v = memref.load %src[%tid] : memref<64xf32>
      memref.store %v, %dst[%tid] : memref<64xf32>
      gpu.barrier
      %w = memref.load %src[%c0] : memref<64xf32>
      memref.store %w, %dst[%c0] : memref<64xf32>
      gpu.return
    }
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
    gpu.launch_func @kernels::@copy
        blocks in (%grid, %grid, %grid)
        threads in (%block, %block, %block)
        args(%src : memref<64xf32>, %dst : memref<64xf32>)
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
