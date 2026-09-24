// ROUGE-V contour: a CUDA-flavoured GPU module in the shape the SIMT->RVV
// lowering will consume. The report pass must classify it as "mixed access —
// vectorize only the runs between barriers" (the GEMM tile shape) and as a
// pure unit-stride candidate for the elementwise copy.

module {
  gpu.module @kernels {
    gpu.func @gemm_tile(%A: memref<128x64xf16>, %B: memref<64x128xf16>,
                        %C: memref<128x128xf32>) kernel {
      %c0 = arith.constant 0 : index
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %a = memref.load %A[%c0, %c0] : memref<128x64xf16>
      %b = memref.load %B[%c0, %c0] : memref<64x128xf16>
      memref.store %a, %C[%c0, %c0] : memref<128x128xf32>
      gpu.barrier
      %acc = memref.atomic_rmw add %C[%c0, %c0], %a : memref<128x128xf32>
      vector.store %acc, %C[%c0, %c0] : memref<128x128xf32>, vector<8xf32>
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
