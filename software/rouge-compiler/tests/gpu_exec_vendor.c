// Tier-1: vendor kernels (genuine nvcc output, kernels/vendor/*.ptx) compiled
// by OUR ptx2ir for the NVPTX backend and executed on a real NVIDIA GPU,
// verified bit-exactly (or within a stated tolerance) against a host reference.
//
// Each cell names its .ptx and entry explicitly: these are multi-entry modules
// exactly like the cuda-samples originals. Usage:
//   gpu_exec_vendor <vendor_dir> <cell> ...
// cells: vecAdd | matrixMul | transpose | reduction | threadFence | all
//
// Failure policy mirrors gpu_exec_rungpu.c: no device or a card older than
// sm_75 prints a SKIP marker and exits 0 (CTest maps the markers to SKIP);
// anything else is a hard failure.

#include <cuda.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_rc = 0;

#define CK(x)                                                                 \
  do {                                                                        \
    CUresult _r = (x);                                                        \
    if (_r != CUDA_SUCCESS) {                                                 \
      const char* _s = nullptr;                                               \
      cuGetErrorString(_r, &_s);                                              \
      printf("  FAIL %s -> %s (%d)\n", #x, _s ? _s : "?", (int)_r);           \
      return 3;                                                               \
    }                                                                         \
  } while (0)

static char* slurp(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return nullptr;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* b = (char*)malloc((size_t)n + 1);
  if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
    free(b);
    fclose(f);
    return nullptr;
  }
  b[n] = 0;
  fclose(f);
  return b;
}

static unsigned rng_state = 0x9E3779B9u;
static float rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return (float)((int)((rng_state >> 8) % 2001) - 1000) / 500.0f;
}

static CUmodule load(const char* dir, const char* file) {
  char path[512];
  snprintf(path, sizeof path, "%s/%s.ptx", dir, file);
  char* txt = slurp(path);
  if (!txt) {
    printf("  FAIL cannot read %s\n", path);
    return nullptr;
  }
  CUmodule mod = nullptr;
  CUresult r = cuModuleLoadData(&mod, txt);
  free(txt);
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    printf("  FAIL load %s -> %s (%d)\n", file, s ? s : "?", (int)r);
    return nullptr;
  }
  return mod;
}

// vecAdd: 50000 elements, bit-exact.
static int cell_vecAdd(const char* dir) {
  CUmodule mod = load(dir, "vecAdd");
  if (!mod) return 3;
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "VecAdd_kernel"));
  const int N = 50000, T = 256, G = (N + T - 1) / T;
  printf("  config: N=%d blockDim=(%d,1,1) gridDim=(%d,1,1)\n", N, T, G);
  float *ha = (float*)malloc(N * 4), *hb = (float*)malloc(N * 4);
  float *hc = (float*)malloc(N * 4), *he = (float*)malloc(N * 4);
  for (int i = 0; i < N; ++i) {
    ha[i] = (float)i * 0.25f;
    hb[i] = (float)(i * 2) - 0.5f;
    he[i] = ha[i] + hb[i];
  }
  memset(hc, 0x5a, N * 4);
  CUdeviceptr da, db, dc;
  CK(cuMemAlloc(&da, N * 4));
  CK(cuMemAlloc(&db, N * 4));
  CK(cuMemAlloc(&dc, N * 4));
  CK(cuMemcpyHtoD(da, ha, N * 4));
  CK(cuMemcpyHtoD(db, hb, N * 4));
  CK(cuMemcpyHtoD(dc, hc, N * 4));
  int n = N;
  void* args[] = {&da, &db, &dc, &n};
  CK(cuLaunchKernel(fn, G, 1, 1, T, 1, 1, 0, 0, args, 0));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(hc, dc, N * 4));
  int bad = 0, first = -1;
  for (int i = 0; i < N; ++i)
    if (memcmp(&hc[i], &he[i], 4)) {
      if (first < 0) first = i;
      ++bad;
    }
  if (bad) {
    printf("  MISMATCH vecAdd: %d/%d wrong, first at %d (got %f want %f)\n", bad, N, first,
           hc[first], he[first]);
    g_rc = 1;
  } else {
    printf("  OK: vecAdd %d elements bit-exact\n", N);
  }
  cuMemFree(da);
  cuMemFree(db);
  cuMemFree(dc);
  cuModuleUnload(mod);
  free(ha);
  free(hb);
  free(hc);
  free(he);
  return 0;
}

// matrixMul: 32x32 via the BLOCK_SIZE 16 template, bit-exact against the same
// sequential fmaf chain.
static int cell_matrixMul(const char* dir) {
  CUmodule mod = load(dir, "matrixMul");
  if (!mod) return 3;
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "_Z13MatrixMulCUDAILi16EEvPfS0_S0_ii"));
  const int SZ = 32, BS = 16, G = SZ / BS;
  printf("  config: %dx%d C=A*B, blockDim=(%d,%d) gridDim=(%d,%d)\n", SZ, SZ, BS, BS, G, G);
  float *hA = (float*)malloc(SZ * SZ * 4), *hB = (float*)malloc(SZ * SZ * 4);
  float *hC = (float*)malloc(SZ * SZ * 4), *hE = (float*)malloc(SZ * SZ * 4);
  for (int i = 0; i < SZ * SZ; ++i) {
    hA[i] = rnd();
    hB[i] = rnd();
  }
  for (int r = 0; r < SZ; ++r)
    for (int c = 0; c < SZ; ++c) {
      float acc = 0.0f;
      for (int k = 0; k < SZ; ++k) acc = fmaf(hA[r * SZ + k], hB[k * SZ + c], acc);
      hE[r * SZ + c] = acc;
    }
  memset(hC, 0x5a, SZ * SZ * 4);
  CUdeviceptr dA, dB, dC;
  CK(cuMemAlloc(&dA, SZ * SZ * 4));
  CK(cuMemAlloc(&dB, SZ * SZ * 4));
  CK(cuMemAlloc(&dC, SZ * SZ * 4));
  CK(cuMemcpyHtoD(dA, hA, SZ * SZ * 4));
  CK(cuMemcpyHtoD(dB, hB, SZ * SZ * 4));
  CK(cuMemcpyHtoD(dC, hC, SZ * SZ * 4));
  int wA = SZ, wB = SZ;
  void* args[] = {&dC, &dA, &dB, &wA, &wB};
  CK(cuLaunchKernel(fn, G, G, 1, BS, BS, 1, 0, 0, args, 0));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(hC, dC, SZ * SZ * 4));
  int bad = 0, first = -1;
  for (int i = 0; i < SZ * SZ; ++i)
    if (memcmp(&hC[i], &hE[i], 4)) {
      if (first < 0) first = i;
      ++bad;
    }
  if (bad) {
    printf("  MISMATCH matrixMul: %d/%d wrong, first at %d (got %f want %f)\n", bad, SZ * SZ,
           first, hC[first], hE[first]);
    g_rc = 1;
  } else {
    printf("  OK: matrixMul %dx%d bit-exact\n", SZ, SZ);
  }
  cuMemFree(dA);
  cuMemFree(dB);
  cuMemFree(dC);
  cuModuleUnload(mod);
  free(hA);
  free(hB);
  free(hC);
  free(hE);
  return 0;
}

// transpose: 128x128 through the padded shared tile, bit-exact.
static int cell_transpose(const char* dir) {
  CUmodule mod = load(dir, "transpose");
  if (!mod) return 3;
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "transposeCoalesced"));
  const int W = 128;
  const int G = (W + 31) / 32;
  printf("  config: %dx%d, blockDim=(32,8) gridDim=(%d,%d)\n", W, W, G, G);
  float* hin = (float*)malloc(W * W * 4);
  float* hout = (float*)malloc(W * W * 4);
  float* hexp = (float*)malloc(W * W * 4);
  for (int i = 0; i < W * W; ++i) hin[i] = (float)i * 0.5f - 3.0f;
  for (int r = 0; r < W; ++r)
    for (int c = 0; c < W; ++c) hexp[c * W + r] = hin[r * W + c];
  memset(hout, 0x5a, W * W * 4);
  CUdeviceptr di, dOut;
  CK(cuMemAlloc(&di, W * W * 4));
  CK(cuMemAlloc(&dOut, W * W * 4));
  CK(cuMemcpyHtoD(di, hin, W * W * 4));
  CK(cuMemcpyHtoD(dOut, hout, W * W * 4));
  int w = W;
  void* args[] = {&dOut, &di, &w};
  CK(cuLaunchKernel(fn, G, G, 1, 32, 8, 1, 0, 0, args, 0));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(hout, dOut, W * W * 4));
  int bad = 0, first = -1;
  for (int i = 0; i < W * W; ++i)
    if (memcmp(&hout[i], &hexp[i], 4)) {
      if (first < 0) first = i;
      ++bad;
    }
  if (bad) {
    printf("  MISMATCH transpose: %d/%d wrong, first at %d (got %f want %f)\n", bad, W * W,
           first, hout[first], hexp[first]);
    g_rc = 1;
  } else {
    printf("  OK: transpose %dx%d bit-exact\n", W, W);
  }
  cuMemFree(di);
  cuMemFree(dOut);
  cuModuleUnload(mod);
  free(hin);
  free(hout);
  free(hexp);
  return 0;
}

// reduction: blockSum tree + atomicAdd. The tree order differs from a
// sequential sum, so the check is a stated relative tolerance.
static int cell_reduction(const char* dir) {
  CUmodule mod = load(dir, "reduction");
  if (!mod) return 3;
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "blockSum"));
  const int N = 4096, T = 256, G = N / T;
  printf("  config: N=%d blockDim=(%d,1,1) gridDim=(%d,1,1)\n", N, T, G);
  float* hin = (float*)malloc(N * 4);
  double ref = 0.0;
  for (int i = 0; i < N; ++i) {
    hin[i] = rnd();
    ref += hin[i];
  }
  CUdeviceptr di, dsum;
  CK(cuMemAlloc(&di, N * 4));
  CK(cuMemAlloc(&dsum, 4));
  CK(cuMemcpyHtoD(di, hin, N * 4));
  float zero = 0.0f;
  CK(cuMemcpyHtoD(dsum, &zero, 4));
  int n = N;
  void* args[] = {&di, &dsum, &n};
  CK(cuLaunchKernel(fn, G, 1, 1, T, 1, 1, 0, 0, args, 0));
  CK(cuCtxSynchronize());
  float got = 0.0f;
  CK(cuMemcpyDtoH(&got, dsum, 4));
  const double err = fabs((double)got - ref) / (fabs(ref) + 1e-6);
  if (!(err < 1e-4)) {
    printf("  MISMATCH reduction: got %.4f ref %.4f relerr %.3e\n", (double)got, ref, err);
    g_rc = 1;
  } else {
    printf("  OK: reduction N=%d sum %.4f (ref %.4f, relerr %.2e)\n", N, (double)got, ref, err);
  }
  cuMemFree(di);
  cuMemFree(dsum);
  cuModuleUnload(mod);
  free(hin);
  return 0;
}

// threadFence: producer/consumer protocol across two launches. Exercises
// membar.gl and atom.global.exch.b32; the consumer's spin loop relies on the
// driver having completed the previous launch.
static int cell_threadFence(const char* dir) {
  CUmodule mod = load(dir, "threadFence");
  if (!mod) return 3;
  CUfunction prod, cons;
  CK(cuModuleGetFunction(&prod, mod, "fenceProducer"));
  CK(cuModuleGetFunction(&cons, mod, "fenceConsumer"));
  const int N = 1024, T = 256, G = N / T;
  printf("  config: N=%d producer grid=(%d,1,1) then consumer (1,1,1)\n", N, G);
  CUdeviceptr ddata, dflag, dout;
  CK(cuMemAlloc(&ddata, N * 4));
  CK(cuMemAlloc(&dflag, 4));
  CK(cuMemAlloc(&dout, 4));
  float* zeros = (float*)calloc(N, 4);
  CK(cuMemcpyHtoD(ddata, zeros, N * 4));
  unsigned int flag0 = 0;
  CK(cuMemcpyHtoD(dflag, &flag0, 4));
  int n = N;
  void* pargs[] = {&ddata, &dflag, &n};
  CK(cuLaunchKernel(prod, G, 1, 1, T, 1, 1, 0, 0, pargs, 0));
  CK(cuCtxSynchronize());
  unsigned int flag = 0;
  CK(cuMemcpyDtoH(&flag, dflag, 4));
  if (flag != 1u) {
    printf("  MISMATCH threadFence: flag=%u, expected 1 (membar.gl + atom.global.exch)\n", flag);
    g_rc = 1;
    cuMemFree(ddata);
    cuMemFree(dflag);
    cuMemFree(dout);
    cuModuleUnload(mod);
    free(zeros);
    return 0;
  }
  double ref = 0.0;
  for (int i = 0; i < N; ++i) ref += (double)((float)i * 1.5f + 0.25f);
  void* cargs[] = {&ddata, &dflag, &dout, &n};
  CK(cuLaunchKernel(cons, 1, 1, 1, 1, 1, 1, 0, 0, cargs, 0));
  CK(cuCtxSynchronize());
  float got = 0.0f;
  CK(cuMemcpyDtoH(&got, dout, 4));
  const double err = fabs((double)got - ref) / (fabs(ref) + 1e-6);
  if (!(err < 1e-4)) {
    printf("  MISMATCH threadFence: got %.4f ref %.4f relerr %.3e\n", (double)got, ref, err);
    g_rc = 1;
  } else {
    printf("  OK: threadFence producer/consumer sum %.4f (ref %.4f, relerr %.2e)\n",
           (double)got, ref, err);
  }
  cuMemFree(ddata);
  cuMemFree(dflag);
  cuMemFree(dout);
  cuModuleUnload(mod);
  free(zeros);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage: %s <vendor_dir> <vecAdd|matrixMul|transpose|reduction|threadFence|all>\n",
           argv[0]);
    return 2;
  }
  const char* dir = argv[1];
  const char* cell = argv[2];

  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) {
    printf("NO-DEVICE: cuInit failed (%d) - skipping\n", (int)r);
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
    printf("SKIP-OLD-ARCH: device sm_%d%d is older than the PTX target sm_75 - skipping\n", major,
           minor);
    return 0;
  }
  printf("== vendor Tier-1 on sm_%d%d ==\n", major, minor);
  CUcontext ctx = nullptr;
  if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    printf("FAIL: cannot make a context current\n");
    return 3;
  }

  int rc = 0;
  if (!strcmp(cell, "vecAdd") || !strcmp(cell, "all")) rc |= cell_vecAdd(dir);
  if (!strcmp(cell, "matrixMul") || !strcmp(cell, "all")) rc |= cell_matrixMul(dir);
  if (!strcmp(cell, "transpose") || !strcmp(cell, "all")) rc |= cell_transpose(dir);
  if (!strcmp(cell, "reduction") || !strcmp(cell, "all")) rc |= cell_reduction(dir);
  if (!strcmp(cell, "threadFence") || !strcmp(cell, "all")) rc |= cell_threadFence(dir);
  cuCtxDestroy(ctx);
  if (g_rc) {
    printf("VENDOR TIER-1: FAILED\n");
    return 1;
  }
  printf("VENDOR TIER-1: all cells passed\n");
  return rc ? 1 : 0;
}