// vadd via the CUDA Driver API on the CPU-emulated ROUGE device.
// The PTX below is hand-written nvcc-style output; this is the exact "legacy CUDA
// code" that the compat layer must run without modifications.

#include <rouge/cuda.h>

#include <cstdio>
#include <cstring>
#include <vector>

static const char* kPtx = R"(
.version 8.0
.target sm_80
.address_size 64

.visible .entry vadd(
    .param .u64 vadd_param_0,
    .param .u64 vadd_param_1,
    .param .u64 vadd_param_2,
    .param .u32 vadd_param_3
)
{
    .reg .b32 %r<7>;
    .reg .b64 %rd<6>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [vadd_param_0];
    ld.param.u64 %rd2, [vadd_param_1];
    ld.param.u64 %rd3, [vadd_param_2];
    ld.param.u32 %r1, [vadd_param_3];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.wide.u32 %rd4, %r2, %r3;
    mov.u32 %r2, %tid.x;
    cvt.u64.u32 %rd5, %r2;
    add.s64 %rd4, %rd4, %rd5;
    cvt.u32.u64 %r4, %rd4;
    setp.ge.u32 %p1, %r4, %r1;
    @%p1 bra LBB0_2;
    mul.wide.u32 %rd5, %r4, 4;
    add.s64 %rd1, %rd1, %rd5;
    add.s64 %rd2, %rd2, %rd5;
    add.s64 %rd3, %rd3, %rd5;
    ld.global.f32 %r5, [%rd1];
    ld.global.f32 %r6, [%rd2];
    add.f32 %r7, %r5, %r6;
    st.global.f32 [%rd3], %r7;
LBB0_2:
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
                   #expr, _s, static_cast<int>(_r));                       \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main() {
  const int N = 4096;
  std::vector<float> a(N), b(N), c(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
  }

  CHECK(cuInit(0));

  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, 0));

  CUdeviceptr da = 0, db = 0, dc = 0;
  CHECK(cuMemAlloc(&da, N * sizeof(float)));
  CHECK(cuMemAlloc(&db, N * sizeof(float)));
  CHECK(cuMemAlloc(&dc, N * sizeof(float)));
  CHECK(cuMemcpyHtoD(da, a.data(), N * sizeof(float)));
  CHECK(cuMemcpyHtoD(db, b.data(), N * sizeof(float)));

  CUmodule mod = nullptr;
  CHECK(cuModuleLoadData(&mod, kPtx));

  CUfunction fn = nullptr;
  CHECK(cuModuleGetFunction(&fn, mod, "vadd"));

  const unsigned int n = static_cast<unsigned int>(N);
  void* params[4] = {&da, &db, &dc, const_cast<unsigned int*>(&n)};
  CHECK(cuLaunchKernel(fn, N / 256, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr));
  CHECK(cuCtxSynchronize());

  CHECK(cuMemcpyDtoH(c.data(), dc, N * sizeof(float)));

  for (int i = 0; i < N; ++i) {
    const float expect = a[i] + b[i];
    if (c[i] != expect) {
      std::fprintf(stderr, "mismatch at %d: got %f, expected %f\n", i, c[i], expect);
      return 1;
    }
  }

  std::printf("PASS: driver-API vector add (N=%d) on CPU-emulated ROUGE device\n", N);

  CHECK(cuMemFree(da));
  CHECK(cuMemFree(db));
  CHECK(cuMemFree(dc));
  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));
  return 0;
}