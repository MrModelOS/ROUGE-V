// Standalone GPU check for the atomic_reduce kernel (separate from rungpu.c).
// Kernel contract, read off atomic_reduce.ptx:
//   params: data ptr (f32[N]), sum ptr (f32[1]), count ptr (u32[1]),
//           slots ptr (u32[N], zero-initialised), n (u32).
//   each in-bounds thread i: sum += data[i] (atomic f32), count += 1,
//   and writes back the old value of slots[i] (must be 0 for every slot).
// Data are multiples of 0.25 and the total stays below 2^24, so the f32 sum
// is exact whatever order the atomics complete in.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// Create a context, retrying once: the driver occasionally answers the first
// cuCtxCreate after a burst of launches with a transient CUDA_ERROR_UNKNOWN.
// One bounded retry with a pause distinguishes that from a real failure; the
// retry is always reported, never silent.
static int make_ctx(CUcontext* ctx, CUdevice dev) {
  CUresult r = cuCtxCreate(ctx, 0, dev, 0);
  if (r != CUDA_SUCCESS) {
    const char* n_;
    cuGetErrorName(r, &n_);
    printf("  (cuCtxCreate retry after %s, pausing 5 s)\n", n_);
    CUresult r2 = cuCtxSynchronize();
    (void)r2;
    struct timespec ts;
    ts.tv_sec = 5;
    ts.tv_nsec = 0;
    nanosleep(&ts, 0);
    r = cuCtxCreate(ctx, 0, dev, 0);
  }
  if (r != CUDA_SUCCESS) {
    const char* n_;
    cuGetErrorName(r, &n_);
    printf("FAIL cuCtxCreate: %s\n", n_);
    return 1;
  }
  return 0;
}

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
  const char* ptx = argc > 1 ? argv[1] : "atomic_reduce.ptx";
  const int N = 1024;
  const int threads = 256;
  const unsigned blocks = (N + threads - 1) / threads;

  FILE* f = fopen(ptx, "rb");
  if (!f) { printf("FAIL open %s\n", ptx); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  char* buf = malloc(sz + 1);
  if (fread(buf, 1, sz, f) != (size_t)sz) { printf("FAIL read\n"); return 1; }
  buf[sz] = 0;
  fclose(f);

  // Environment guards: this test needs a real NVIDIA GPU on the host.
  // Without one (CI machines, AMD/Intel-only boxes) it reports SKIP instead
  // of failing. CTest matches these markers via SKIP_REGULAR_EXPRESSION.
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
    // The PTX below is built for sm_75; the driver JIT cannot retarget it to
    // an older architecture, so such a card is an environment limitation.
    if (ccmaj < 7 || (ccmaj == 7 && ccmin < 5)) {
      printf("SKIP-OLD-ARCH: device cc %d.%d < 7.5, PTX targets sm_75\n", ccmaj, ccmin);
      return 0;
    }
  }
  CUcontext ctx;
  if (make_ctx(&ctx, dev)) return 1;
  CUmodule mod;
  CK(cuModuleLoadData(&mod, buf));
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "atomic_reduce"));

  CUdeviceptr d_data, d_sum, d_count, d_slots;
  CK(cuMemAlloc(&d_data, N * 4));
  CK(cuMemAlloc(&d_sum, 4));
  CK(cuMemAlloc(&d_count, 4));
  CK(cuMemAlloc(&d_slots, N * 4));

  float* h_data = malloc(N * 4);
  unsigned* h_slots = calloc(N, 4);
  float want_sum = 0;
  for (int i = 0; i < N; ++i) {
    h_data[i] = (float)((i % 16) * 0.25f);
    want_sum += h_data[i];
  }
  float h_sum = 0;
  unsigned h_count = 0;
  CK(cuMemcpyHtoD(d_data, h_data, N * 4));
  CK(cuMemcpyHtoD(d_sum, &h_sum, 4));
  CK(cuMemcpyHtoD(d_count, &h_count, 4));
  CK(cuMemcpyHtoD(d_slots, h_slots, N * 4));

  int n = N;
  void* args[] = {&d_data, &d_sum, &d_count, &d_slots, &n};
  CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(&h_sum, d_sum, 4));
  CK(cuMemcpyDtoH(&h_count, d_count, 4));
  CK(cuMemcpyDtoH(h_slots, d_slots, N * 4));

  int bad = 0;
  if (h_sum != want_sum) {
    printf("  sum: got %g want %g\n", h_sum, want_sum);
    bad = 1;
  }
  if (h_count != (unsigned)N) {
    printf("  count: got %u want %d\n", h_count, N);
    bad = 1;
  }
  int bad_slots = 0;
  for (int i = 0; i < N; ++i)
    if (h_slots[i] != 0 && bad_slots++ < 4)
      printf("  slot[%d] = %u, want 0\n", i, h_slots[i]);
  if (bad_slots) { printf("  %d slots nonzero\n", bad_slots); bad = 1; }
  printf(bad ? "MISMATCH\n" : "OK: sum=%g count=%u slots all zero (%d threads)\n",
         h_sum, h_count, N);
  return bad;
}
