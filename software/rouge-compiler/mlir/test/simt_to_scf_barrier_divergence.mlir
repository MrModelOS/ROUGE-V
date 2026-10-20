// ROUGE-V contour: a BARRIER inside a divergent branch — must be REFUSED.
//
// The gpu.barrier contract: once one thread of a scope has executed a
// particular dynamic instance of gpu.barrier, all other threads of that scope
// must execute the same dynamic instance of it before any thread executes any
// other instance. A barrier inside a branch that only some threads take
// breaks that contract, so the GPU's result is undefined there.
//
// Our phase decomposition (the body is split at barriers so that every
// thread of a phase runs before the next one) assumes every thread reaches
// every barrier. That assumption is exactly what breaks here, so the pass
// must REFUSE loudly instead of producing IR that looks fine and computes
// something else.
//
// NOTE: the file must be valid MLIR — the pass verifies the module before
// rewriting it, so a type error here is a hard failure, not a warning.

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @barrier_in_divergence(%dst: memref<16xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %cmp = arith.cmpi sgt, %tid, %c1 : index
      %v = memref.load %dst[%c0] : memref<16xf32>
      scf.if %cmp {
        memref.store %v, %dst[%tid] : memref<16xf32>
        gpu.barrier
      }
      gpu.return
    }
  }
}
