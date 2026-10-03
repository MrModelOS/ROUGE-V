// Device queries + cuModuleGetGlobal coherence on the CPU backend.
// 1. cuDriverGetVersion / cuDeviceGetName / cuDeviceComputeCapability /
//    cuDeviceGetAttribute / cuDeviceTotalMem / cuCtxSetCurrent+GetCurrent /
//    cuCtxGetDevice / cuMemcpyDtoD / cuMemHostAlloc+GetDevicePointer /
//    cuStreamCreate+Synchronize+Query+Destroy / cuFuncGetAttribute /
//    cuGetErrorName answer the documented contract values.
// 2. A host write through a cuModuleGetGlobal pointer is visible to a
//    subsequently launched kernel (module-global backing store shared
//    with the interpreter): the kernel loads a .global float and adds it
//    to every output element.

#include <rouge/cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

// Kernel with a module-scope .global: nvcc emits exactly this shape for a
// __device__ variable read through cvta.to.global.
static const char* kPtx = R"(
.version 8.0
.target sm_75
.address_size 64

.global .align 4 .f32 g_bias;

.visible .entry gbias(
    .param .u64 gbias_param_0,
    .param .u64 gbias_param_1,
    .param .u32 gbias_param_2
)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<6>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [gbias_param_0];
    ld.param.u64 %rd2, [gbias_param_1];
    ld.param.u32 %r1, [gbias_param_2];

    mov.u32 %r2, %tid.x;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L_done;

    mul.wide.u32 %rd3, %r2, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %r3, [%rd4];
    mov.b64 %rd6, g_bias;
    ld.global.f32 %r4, [%rd6];
    add.f32 %r5, %r3, %r4;
    st.global.f32 [%rd5], %r5;
$L_done:
    ret;
}
)";

#define CHECK(expr)                                                        \
  do {                                                                     \
    const CUresult _r = (expr);                                            \
    if (_r != CUDA_SUCCESS) {                                              \
      const char* _s = "?";                                                \
      cuGetErrorString(_r, &_s);                                           \
      std::fprintf(stderr, "%s:%d: %s -> %s (%d)\n", __FILE__, __LINE__,   \
                   #expr, _s, (int)_r);                                    \
      return 1;                                                            \
    }                                                                      \
  } while (0)

#define EXPECT(cond)                                                       \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "%s:%d: EXPECT failed: %s\n", __FILE__,         \
                   __LINE__, #cond);                                       \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main() {
  CHECK(cuInit(0));

  // ---- device / context queries ----
  int ver = 0;
  CHECK(cuDriverGetVersion(&ver));
  EXPECT(ver == 13040);
  char name[64];
  CHECK(cuDeviceGetName(name, sizeof name, 0));
  EXPECT(std::strlen(name) > 0);
  int major = 0, minor = 0;
  CHECK(cuDeviceComputeCapability(&major, &minor, 0));
  EXPECT(major == 7 && minor == 5);
  int warp = 0, mpCount = 0;
  CHECK(cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0));
  EXPECT(warp == 32);
  CHECK(cuDeviceGetAttribute(&mpCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0));
  EXPECT(mpCount == 1);
  // Undeclared attributes refuse loudly, never guess.
  EXPECT(cuDeviceGetAttribute(&warp, (CUdevice_attribute)9999, 0) ==
         CUDA_ERROR_NOT_SUPPORTED);
  size_t total = 0;
  CHECK(cuDeviceTotalMem(&total, 0));
  EXPECT(total > 0);
  size_t freeMem = 0, totalMem = 0;
  CHECK(cuMemGetInfo(&freeMem, &totalMem));
  EXPECT(totalMem == total && freeMem == total);

  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, 0));
  CHECK(cuCtxSetCurrent(ctx));
  CUcontext cur = nullptr;
  CHECK(cuCtxGetCurrent(&cur));
  EXPECT(cur == ctx);
  CUdevice dev = -1;
  CHECK(cuCtxGetDevice(&dev));
  EXPECT(dev == 0);

  // ---- error names ----
  const char* es = nullptr;
  CHECK(cuGetErrorName(CUDA_SUCCESS, &es));
  EXPECT(std::strcmp(es, "CUDA_SUCCESS") == 0);
  CHECK(cuGetErrorName(CUDA_ERROR_NOT_SUPPORTED, &es));
  EXPECT(std::strcmp(es, "CUDA_ERROR_NOT_SUPPORTED") == 0);
  CHECK(cuGetErrorString(CUDA_ERROR_NOT_SUPPORTED, &es));

  // ---- streams are synchronous tokens ----
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, 0));
  EXPECT(stream != nullptr);
  CHECK(cuStreamQuery(stream));
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuStreamDestroy(stream));

  // ---- pinned host memory + device pointer identity ----
  void* pinned = nullptr;
  CHECK(cuMemHostAlloc(&pinned, 64, 0));
  EXPECT(pinned != nullptr);
  CUdeviceptr dPinned = 0;
  CHECK(cuMemHostGetDevicePointer(&dPinned, pinned, 0));
  EXPECT(dPinned == static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(pinned)));
  CHECK(cuMemFreeHost(pinned));

  // ---- module global roundtrip ----
  CUmodule mod = nullptr;
  CHECK(cuModuleLoadData(&mod, kPtx));
  CUfunction fn = nullptr;
  CHECK(cuModuleGetFunction(&fn, mod, "gbias"));
  int sharedBytes = -1, maxThreads = -1;
  CHECK(cuFuncGetAttribute(&sharedBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
  EXPECT(sharedBytes == 0);
  CHECK(cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn));
  EXPECT(maxThreads == 1024);
  // NUM_REGS needs a register allocator: loud refusal.
  EXPECT(cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_NUM_REGS, fn) ==
         CUDA_ERROR_NOT_SUPPORTED);

  CUdeviceptr dBias = 0;
  size_t biasBytes = 0;
  CHECK(cuModuleGetGlobal(&dBias, &biasBytes, mod, "g_bias"));
  EXPECT(dBias != 0 && biasBytes == 4);
  EXPECT(cuModuleGetGlobal(&dBias, nullptr, mod, "no_such_symbol") ==
         CUDA_ERROR_INVALID_VALUE);
  const float bias = 2.5f;
  CHECK(cuMemcpyHtoD(dBias, &bias, sizeof bias));

  const int N = 256;
  float hostIn[N], hostOut[N];
  for (int i = 0; i < N; ++i) {
    hostIn[i] = (float)i * 0.5f;
    hostOut[i] = 0.0f;
  }
  CUdeviceptr dIn = 0, dOut = 0;
  CHECK(cuMemAlloc(&dIn, sizeof hostIn));
  CHECK(cuMemAlloc(&dOut, sizeof hostOut));
  CHECK(cuMemcpyHtoD(dIn, hostIn, sizeof hostIn));
  // Device-to-device copy exercises cuMemcpyDtoD.
  CHECK(cuMemcpyDtoD(dOut, dIn, sizeof hostIn));

  CUdeviceptr inArg = dIn, outArg = dOut;
  int nArg = N;
  void* args[] = {&inArg, &outArg, &nArg};
  CHECK(cuLaunchKernel(fn, 1, 1, 1, (unsigned)N, 1, 1, 0, nullptr, args, nullptr));
  CHECK(cuCtxSynchronize());
  CHECK(cuMemcpyDtoH(hostOut, dOut, sizeof hostOut));
  for (int i = 0; i < N; ++i) {
    const float want = hostIn[i] + bias;
    if (hostOut[i] != want) {
      std::fprintf(stderr, "i=%d got=%f want=%f\n", i, hostOut[i], want);
      return 1;
    }
  }

  CHECK(cuMemFree(dIn));
  CHECK(cuMemFree(dOut));
  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));
  std::printf("PASS: device queries + cuModuleGetGlobal coherence on CPU backend\n");
  return 0;
}
