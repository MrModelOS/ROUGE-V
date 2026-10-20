// ROUGE-V differential fuzzer: the ARBITER side (the real NVIDIA card).
//
// Takes the PTX that OUR ptx2ir produced (not the original one — the point
// is to test the whole chain: our compiler -> the card), the input data, and
// writes the card's output. The interpreter runner (run_interp.cpp) takes the
// SAME original kernel and writes the reference output; the driver compares.
//
// The card is the arbiter: where our belief about PTX semantics (the
// interpreter) and the hardware disagree, the hardware is right — that is how
// the prmt selector-field bug was found.
//
// usage: run_gpu <kernel.ptx> <in.bin> <out.bin> <n>

#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x)                                                                 \
  do {                                                                        \
    CUresult _r = (x);                                                        \
    if (_r != CUDA_SUCCESS) {                                                 \
      const char* _s = NULL;                                                  \
      cuGetErrorString(_r, &_s);                                              \
      printf("  FAIL %s -> %s (%d)\n", #x, _s ? _s : "?", (int)_r);           \
      return 3;                                                               \
    }                                                                         \
  } while (0)

static char* slurp(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* b = (char*)malloc((size_t)n + 1);
  if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
    free(b);
    fclose(f);
    return NULL;
  }
  b[n] = 0;
  fclose(f);
  return b;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    printf("usage: %s <kernel.ptx> <in.bin> <out.bin> <n>\n", argv[0]);
    return 2;
  }
  const char* ptxPath = argv[1];
  const char* inPath = argv[2];
  const char* outPath = argv[3];
  const int n = atoi(argv[4]);
  const int kOut = 4;

  if (cuInit(0) != CUDA_SUCCESS) {
    printf("NO-DEVICE: cuInit failed - skipping\n");
    return 0;
  }
  int count = 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count == 0) {
    printf("NO-DEVICE: no CUDA device - skipping\n");
    return 0;
  }
  CUdevice dev = 0;
  int major = 0, minor = 0;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) !=
          CUDA_SUCCESS ||
      cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) !=
          CUDA_SUCCESS) {
    printf("NO-DEVICE: cannot query device - skipping\n");
    return 0;
  }
  if (major * 10 + minor < 75) {
    printf("SKIP-OLD-ARCH: device sm_%d%d is older than sm_75 - skipping\n", major,
           minor);
    return 0;
  }
  CUcontext ctx = NULL;
  if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    printf("FAIL: cannot make a context current\n");
    return 3;
  }

  char* txt = slurp(ptxPath);
  if (!txt) {
    printf("FAIL: cannot read %s\n", ptxPath);
    return 3;
  }
  CUmodule mod = NULL;
  CUresult mr = cuModuleLoadData(&mod, txt);
  free(txt);
  if (mr != CUDA_SUCCESS) {
    const char* s = NULL;
    cuGetErrorString(mr, &s);
    printf("FAIL load %s -> %s (%d)\n", ptxPath, s ? s : "?", (int)mr);
    return 3;
  }
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "fuzz_kernel"));

  int* input = (int*)malloc(sizeof(int) * (size_t)n);
  int* output = (int*)malloc(sizeof(int) * (size_t)n * kOut);
  FILE* f = fopen(inPath, "rb");
  if (!f || fread(input, sizeof(int), (size_t)n, f) != (size_t)n) {
    printf("FAIL: cannot read %s\n", inPath);
    return 3;
  }
  fclose(f);
  memset(output, 0, sizeof(int) * (size_t)n * kOut);

  CUdeviceptr dOut, dIn;
  CK(cuMemAlloc(&dOut, sizeof(int) * (size_t)n * kOut));
  CK(cuMemAlloc(&dIn, sizeof(int) * (size_t)n));
  CK(cuMemcpyHtoD(dIn, input, sizeof(int) * (size_t)n));
  CK(cuMemcpyHtoD(dOut, output, sizeof(int) * (size_t)n * kOut));

  unsigned nn = (unsigned)n;
  unsigned seed = 12345u;
  void* args[] = {&dOut, &dIn, &nn, &seed};
  CK(cuLaunchKernel(fn, 1, 1, 1, (unsigned)n, 1, 1, 0, 0, args, 0));
  CUresult r = cuCtxSynchronize();
  if (r != CUDA_SUCCESS) {
    const char* s = NULL;
    cuGetErrorString(r, &s);
    printf("FAIL launch/sync -> %s (%d)\n", s ? s : "?", (int)r);
    return 3;
  }
  CK(cuMemcpyDtoH(output, dOut, sizeof(int) * (size_t)n * kOut));

  f = fopen(outPath, "wb");
  if (!f) {
    printf("FAIL: cannot write %s\n", outPath);
    return 3;
  }
  fwrite(output, sizeof(int), (size_t)n * kOut, f);
  fclose(f);

  cuMemFree(dOut);
  cuMemFree(dIn);
  cuModuleUnload(mod);
  free(input);
  free(output);
  return 0;
}
