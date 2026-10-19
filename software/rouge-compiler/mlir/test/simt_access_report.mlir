// ROUGE-V contour: a CUDA-flavoured GPU module in the shape the SIMT->RVV
// lowering will consume. The report pass must classify the GEMM tile shape as
// "mixed access — vectorize only the runs between barriers" (it has a barrier)
// and the elementwise copy as a pure unit-stride candidate.
//
// NOTE: the file must be valid MLIR — the pass verifies the module before
// analysing it, so a type error here is a hard failure, not a warning. The
// syntax used here is deliberately version-portable (MLIR 19..23): the
// memref.atomic_rmw spelling changed in MLIR 23, so atomics live in a
// separate file (simt_access_report_atomic.mlir) used only on MLIR >= 23.

module {
  gpu.module @kernels {
    gpu.func @gemm_tile(%A: memref<128x64xf32>, %B: memref<64x128xf32>,
                        %C: memref<128x128xf32>) kernel {
      %c0 = arith.constant 0 : index
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %a = memref.load %A[%c0, %c0] : memref<128x64xf32>
      %b = memref.load %B[%c0, %c0] : memref<64x128xf32>
      memref.store %a, %C[%c0, %c0] : memref<128x128xf32>
      gpu.barrier
      %v = vector.broadcast %a : f32 to vector<8xf32>
      vector.store %v, %C[%c0, %c0] : memref<128x128xf32>, vector<8xf32>
      gpu.return
    }

    gpu.func @scale_copy(%src: memref<1024xf32>, %dst: memref<1024xf32>) kernel {
      %c0 = arith.constant 0 : index
      %v = memref.load %src[%c0] : memref<1024xf32>
      memref.store %v, %dst[%c0] : memref<1024xf32>
      gpu.return
    }
  }
}
