// ROUGE-V contour: the atomic path of the report pass.
//
// memref.atomic_rmw changed its spelling in MLIR 23 (the kind became an
// attribute and the value operand moved first), so this file is used only
// when the MLIR in use is >= 23. On older MLIR the same detection is covered
// by the portable file simt_access_report.mlir (barrier -> "mixed access").

module {
  gpu.module @kernels {
    gpu.func @atomic_update(%C: memref<128x128xf32>, %a: f32) kernel {
      %c0 = arith.constant 0 : index
      %acc = memref.atomic_rmw "addf" %a, %C[%c0, %c0] : (f32, memref<128x128xf32>) -> f32
      memref.store %acc, %C[%c0, %c0] : memref<128x128xf32>
      gpu.return
    }
  }
}
