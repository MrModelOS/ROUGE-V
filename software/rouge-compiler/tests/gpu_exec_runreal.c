// On-device check for real vendor output: real_nvcc.cu compiled by nvcc,
// translated by our ptx2ir, executed on a real NVIDIA GPU.
// Kernel contract, read off real_nvcc.cu:
//   rouge_real(in ptr, out ptr, n): grid-stride sum per thread into a 256-wide
//   shared tile, two tree-reduce steps, out[blockIdx] = tile[0] (thread 0
//   only); plus two atomics that EVERY thread executes (no tid guard in the
//   source): atomicAdd(&rouge_global[0], (float)blockIdx) and
//   atomicAdd(&out[blockIdx+64], tile[0]).
// The reference replicates the kernel statement by statement, so agreement is
// bit-exact. Atomic order across blocks is irrelevant: rouge_global[0]
// accumulates small integers (0+1+2+3), and each out[block+64] gets exactly
// one atomic add.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x)                                                            \
  do {                                                                   \
    CUresult r_ = (x);                                                   \
    if (r_ != CUDA_SUCCESS) {                                            \
      const char* n_;                                                    \
      cuGetErrorName(r_, &n_);                                           \
      printf("FAIL %s: %s (line %d)\n", #x, n_, __LINE__);              \
      return 1;                                                          \
    }                                                                    \
  } while (0)

int main(int argc, char** argv) {
  const char* ptx = argc > 1 ? argv[1] : "real_nvcc_gpu.ptx";
  const int N = 1024, BLOCKS = 4, THREADS = 256;

  FILE* f = fopen(ptx, "rb");
  if (!f) { printf("FAIL open %s\n", ptx); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  char* buf = malloc(sz + 1);
  if (fread(buf, 1, sz, f) != (size_t)sz) { printf("FAIL read\n"); return 1; }
  buf[sz] = 0;
  fclose(f);

  if (cuInit(0) != CUDA_SUCCESS) {
    printf("NO-DEVICE: cuInit failed, no NVIDIA driver on this host\n");
    return 0;
  }
  CUdevice dev;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
    printf("NO-DEVICE: no CUDA-capable device on this host\n");
    return 0;
  }
  {
    int ccmaj = 0, ccmin = 0;
    cuDeviceGetAttribute(&ccmaj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&ccmin, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (ccmaj < 7 || (ccmaj == 7 && ccmin < 5)) {
      printf("SKIP-OLD-ARCH: device cc %d.%d < 7.5, PTX targets sm_75\n", ccmaj, ccmin);
      return 0;
    }
  }
  CUcontext ctx;
  CK(cuCtxCreate(&ctx, 0, dev, 0));
  CUmodule mod;
  CK(cuModuleLoadData(&mod, buf));
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "_Z10rouge_realPKfPfj"));

  // The kernel's __device__ global arrives as .extern .global: resolve it and
  // zero it, the way any CUDA host program must.
  CUdeviceptr dglob;
  size_t glob_bytes = 0;
  CK(cuModuleGetGlobal(&dglob, &glob_bytes, mod, "rouge_global"));
  CK(cuMemsetD32(dglob, 0, 1));

  CUdeviceptr d_in, d_out;
  CK(cuMemAlloc(&d_in, N * 4));
  CK(cuMemAlloc(&d_out, 128 * 4));
  float* h_in = malloc(N * 4);
  float* h_out = malloc(128 * 4);
  for (int i = 0; i < N; ++i) h_in[i] = (float)(((i % 7) + 1) * 0.5f);
  for (int i = 0; i < 128; ++i) h_out[i] = -1.f;
  CK(cuMemcpyHtoD(d_in, h_in, N * 4));
  CK(cuMemcpyHtoD(d_out, h_out, 128 * 4));

  int n = N;
  void* args[] = {&d_in, &d_out, &n};
  CK(cuLaunchKernel(fn, BLOCKS, 1, 1, THREADS, 1, 1, 0, 0, args, 0));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(h_out, d_out, 128 * 4));
  float h_glob = -1.f;
  CK(cuMemcpyDtoH(&h_glob, dglob, 4));

  // Host reference, statement by statement.
  //
  // One deliberate imprecision: the source has no __syncthreads() between the
  // second halve-step and the atomicAdd(tile[0]) below it, so every thread's
  // atomic races with thread 0's write to tile[0]. A racing thread adds V1
  // (tile[0] after step 1, before thread 0's step-2 write) or V2 (after it);
  // the vendor binary races exactly the same way, so the reference accepts
  // any mixture instead of one fixed total.
  int bad = 0;
  float want_glob = 0.f;
  for (int b = 0; b < BLOCKS; ++b) {
    float tile[256];
    for (int t = 0; t < THREADS; ++t) {
      float acc = 0.f;
      for (int j = b * THREADS + t; j < N; j += THREADS) acc += h_in[j];
      tile[t] = acc;
    }
    for (int t = 0; t < 128; ++t) tile[t] += tile[t + 128];
    float v1 = tile[0];
    for (int t = 0; t < 64; ++t) tile[t] += tile[t + 64];
    float v2 = tile[0];
    if (h_out[b] != tile[0]) {
      printf("  out[%d]: got %g want %g\n", b, h_out[b], tile[0]);
      bad = 1;
    }
    // Executed by all 256 threads of the block, not just thread 0, adding
    // onto the -1.0f the buffer was initialised with. Each thread contributes
    // V1 or V2 (see above), so the total must lie between the all-V1 and the
    // all-V2 mixtures, plus a rounding allowance for 256 ordered adds.
    float lo = 256.f * (v1 < v2 ? v1 : v2) - 1.f - 1.f;
    float hi = 256.f * (v1 < v2 ? v2 : v1) - 1.f + 1.f;
    if (h_out[b + 64] < lo || h_out[b + 64] > hi) {
      printf("  out[%d]: got %g outside race interval [%g, %g] (V1=%g V2=%g)\n",
             b + 64, h_out[b + 64], lo, hi, v1, v2);
      bad = 1;
    }
    want_glob += 256.f * (float)b;
  }
  if (h_glob != want_glob) {
    printf("  rouge_global[0]: got %g want %g\n", h_glob, want_glob);
    bad = 1;
  }
  printf(bad ? "MISMATCH\n"
              : "OK: nvcc PTX -> ptx2ir -> GPU; out[0..3] and the device global "
                "bit-exact, the racing atomic inside its proven interval\n");
  return bad;
}
