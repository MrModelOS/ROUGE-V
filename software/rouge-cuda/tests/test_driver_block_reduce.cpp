// block_reduce (shared memory + bar.sync) via the CUDA Driver API on the
// CPU-emulated ROUGE device — the interpreter fallback path.
// The kernel is byte-identical with the rouge-compiler AOT test (generated
// header, see embed_ptx.cmake): the interpreter must run the same PTX that
// ptx2ir compiles AOT. In emulation threads of a block execute sequentially,
// so bar.sync is trivially satisfied; the shared-memory scratchpad is
// allocated per block by execute_kernel.

#include <rouge/cuda.h>

#include "block_reduce_ptx.h"

#include <cstdio>
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
  const int N = 4096;
  const int BX = 256;  // threads per block
  const int GX = N / BX;
  std::vector<float> data(N);
  std::vector<float> out(static_cast<size_t>(GX), 0.0f);
  for (int i = 0; i < N; ++i) data[i] = static_cast<float>(i + 1);

  CHECK(cuInit(0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, 0));

  CUdeviceptr ddata = 0, dout = 0;
  CHECK(cuMemAlloc(&ddata, N * sizeof(float)));
  CHECK(cuMemAlloc(&dout, static_cast<size_t>(GX) * sizeof(float)));
  CHECK(cuMemcpyHtoD(ddata, data.data(), N * sizeof(float)));

  CUmodule mod = nullptr;
  CHECK(cuModuleLoadData(&mod, kBlockReducePtx));

  CUfunction fn = nullptr;
  CHECK(cuModuleGetFunction(&fn, mod, "block_reduce"));

  const unsigned int n = static_cast<unsigned int>(N);
  void* params[3] = {&ddata, &dout, const_cast<unsigned int*>(&n)};
  CHECK(cuLaunchKernel(fn, GX, 1, 1, BX, 1, 1, 0, nullptr, params, nullptr));
  CHECK(cuCtxSynchronize());

  CHECK(cuMemcpyDtoH(out.data(), dout, static_cast<size_t>(GX) * sizeof(float)));

  for (int b = 0; b < GX; ++b) {
    float expected = 0.0f;
    for (int i = b * BX; i < (b + 1) * BX; ++i) expected += data[i];
    if (out[static_cast<size_t>(b)] != expected) {
      std::fprintf(stderr, "block %d: got %.6f, expected %.6f\n", b,
                   out[static_cast<size_t>(b)], expected);
      return 1;
    }
  }

  std::printf(
      "PASS: driver-API block_reduce (shared memory + bar.sync) on "
      "CPU-emulated ROUGE device (%d blocks x %d threads)\n",
      GX, BX);

  CHECK(cuMemFree(ddata));
  CHECK(cuMemFree(dout));
  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));
  return 0;
}