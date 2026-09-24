// atomic_reduce (relaxed global atomics) via the CUDA Driver API on the
// CPU-emulated ROUGE device — the interpreter fallback path.
//
// The kernel is byte-identical with the rouge-compiler AOT test (generated
// header, see embed_ptx.cmake): both execution paths must agree on the result.
// In the emulator threads of a block execute sequentially, so the atomics are
// trivially satisfied; the AOT path runs them as real hardware atomics on the
// same values, and both must produce the same sum/count/flags.

#include <rouge/cuda.h>

#include "atomic_reduce_ptx.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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
  const uint32_t N = 4096;
  const int BX = 256;
  const int GX = static_cast<int>(N) / BX;

  std::vector<float> data(N);
  float expected = 0.0f;
  for (uint32_t i = 0; i < N; ++i) {
    data[i] = static_cast<float>(i % 8 + 1) * 0.25f;
    expected += data[i];
  }

  CHECK(cuInit(0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, 0));

  CUdeviceptr ddata = 0, dsum = 0, dcount = 0, dflags = 0;
  CHECK(cuMemAlloc(&ddata, N * sizeof(float)));
  CHECK(cuMemAlloc(&dsum, sizeof(float)));
  CHECK(cuMemAlloc(&dcount, sizeof(uint32_t)));
  CHECK(cuMemAlloc(&dflags, N * sizeof(uint32_t)));
  CHECK(cuMemcpyHtoD(ddata, data.data(), N * sizeof(float)));
  const float zero_f = 0.0f;
  const uint32_t zero_u = 0;
  CHECK(cuMemcpyHtoD(dsum, &zero_f, sizeof(float)));
  CHECK(cuMemcpyHtoD(dcount, &zero_u, sizeof(uint32_t)));
  CHECK(cuMemcpyHtoD(dflags, &zero_u, sizeof(uint32_t)));  // claim-and-clear

  CUmodule mod = nullptr;
  CHECK(cuModuleLoadData(&mod, kAtomicReducePtx));
  CUfunction fn = nullptr;
  CHECK(cuModuleGetFunction(&fn, mod, "atomic_reduce"));

  const unsigned int n = N;
  void* params[5] = {&ddata, &dsum, &dcount, &dflags,
                     const_cast<unsigned int*>(&n)};
  CHECK(cuLaunchKernel(fn, GX, 1, 1, BX, 1, 1, 0, nullptr, params, nullptr));
  CHECK(cuCtxSynchronize());

  float sum = 0.0f;
  uint32_t count = 0;
  std::vector<uint32_t> flags(N, 1);
  CHECK(cuMemcpyDtoH(&sum, dsum, sizeof(float)));
  CHECK(cuMemcpyDtoH(&count, dcount, sizeof(uint32_t)));
  CHECK(cuMemcpyDtoH(flags.data(), dflags, N * sizeof(uint32_t)));

  if (sum != expected) {
    std::fprintf(stderr, "sum: got %.2f, expected %.2f\n", sum, expected);
    return 1;
  }
  if (count != N) {
    std::fprintf(stderr, "count: got %u, expected %u\n", count, N);
    return 1;
  }
  for (uint32_t i = 0; i < N; ++i) {
    if (flags[i] != 0) {  // atom.add must return the *old* value (0)
      std::fprintf(stderr, "flags[%u] = %u, expected 0\n", i, flags[i]);
      return 1;
    }
  }

  std::printf(
      "PASS: driver-API atomic_reduce (atom.add.f32/u32, red.add.f32/u32) "
      "on CPU-emulated ROUGE device: sum=%.1f count=%u (%d blocks x %d)\n",
      sum, count, GX, BX);

  CHECK(cuMemFree(ddata));
  CHECK(cuMemFree(dsum));
  CHECK(cuMemFree(dcount));
  CHECK(cuMemFree(dflags));
  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));
  return 0;
}
