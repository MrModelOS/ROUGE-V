// run_on_gpu.c — minimal template: load GPU PTX from `rouge-run --emit-ptx`
// and launch one kernel through the CUDA Driver API.
//
// Flow:
//   ptx2ir --target nvptx64-nvidia-cuda kernel.ptx kernel.ll   (or ptx2ir)
//   rouge-run --target nvptx64-nvidia-cuda kernel.ptx --emit-ptx kernel_gpu.ptx
//   gcc run_on_gpu.c -o run_on_gpu -l:libcuda.so.1
//   ./run_on_gpu
//
// Only the EDIT points below are kernel-specific. Everything else is
// boilerplate: read PTX, create context, load module, launch, sync.
// Needs an sm_75+ NVIDIA GPU and the CUDA driver (libcuda), NOT the
// whole toolkit: nvcc is only needed earlier, to produce the input PTX.
#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>

#define CK(x)                                                  \
  do {                                                         \
    CUresult r_ = (x);                                         \
    if (r_ != CUDA_SUCCESS) {                                  \
      const char* n_ = "?";                                    \
      cuGetErrorName(r_, &n_);                                 \
      printf("FAIL %s: %s\n", #x, n_);                         \
      return 1;                                                \
    }                                                          \
  } while (0)

static char* slurp(const char* path, long* out_sz) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  const long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* buf = (char*)malloc((size_t)sz + 1);
  if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    free(buf);
    fclose(f);
    return NULL;
  }
  buf[sz] = '\0';
  fclose(f);
  if (out_sz) *out_sz = sz;
  return buf;
}

int main(void) {
  /* ---- EDIT 1: PTX file from rouge-run --emit-ptx, mangled kernel name ---- */
  const char* ptx_path = "kernel_gpu.ptx";
  const char* kernel_name = "_Z6vecAddPfS_S_i"; /* read from .visible .entry */
  /* ---- EDIT 2: launch geometry ---- */
  const unsigned blocks = 196, threads = 256;

  char* ptx = slurp(ptx_path, NULL);
  if (!ptx) {
    printf("cannot read %s (run rouge-run --emit-ptx first)\n", ptx_path);
    return 1;
  }

  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CK(cuCtxCreate(&ctx, 0, dev, 0));
  CUmodule mod;
  CK(cuModuleLoadData(&mod, ptx));
  free(ptx);
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, kernel_name));

  /* ---- EDIT 3: device buffers, host data, kernel args, check ----
   * Example for vecAdd(float *A, float *B, float *C, int n):
   *
   *   const int N = 50000;
   *   CUdeviceptr dA, dB, dC;
   *   CK(cuMemAlloc(&dA, N * 4)); CK(cuMemAlloc(&dB, N * 4));
   *   CK(cuMemAlloc(&dC, N * 4));
   *   ... fill host arrays, cuMemcpyHtoD ...
   *   int n = N;
   *   void* args[] = { &dA, &dB, &dC, &n };
   *   CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
   *   CK(cuCtxSynchronize());
   *   ... cuMemcpyDtoH, compare against a host reference ...
   */
  (void)blocks;
  (void)threads;
  (void)fn;

  printf("template loaded '%s' from %s; fill EDIT 3 and rebuild\n",
         kernel_name, ptx_path);
  return 0;
}
