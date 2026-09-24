// fp16_reduce (FP16/BF16 pipeline over a shared-memory reduction) via the CUDA
// Driver API on the CPU-emulated ROUGE device — the interpreter fallback path.
//
// The kernel is byte-identical with the rouge-compiler AOT test. Both paths
// must produce the same 16-bit patterns: the interpreter converts f16/bf16 in
// software (RNE), ptx2ir emits LLVM half/bfloat with fpext/fptrunc.

#include <rouge/cuda.h>

#include "fp16_reduce_ptx.h"

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

uint16_t f32_to_bf16_rne(float f) {  // same RNE rule as LLVM fptrunc -> bfloat
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if (((x >> 23) & 0xffu) == 0xffu) return static_cast<uint16_t>(x >> 16);
  return static_cast<uint16_t>((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}

int main() {
  const uint32_t N = 1024;
  const int BX = 32;
  const int GX = static_cast<int>(N) / BX;

  std::vector<uint16_t> data(N);
  for (uint32_t i = 0; i < N; ++i) {
    _Float16 h = static_cast<_Float16>(static_cast<float>(i % 4 + 1));
    std::memcpy(&data[i], &h, 2);
  }
  std::vector<uint16_t> out_f16(static_cast<size_t>(GX), 0);
  std::vector<uint16_t> out_bf16(static_cast<size_t>(GX), 0);

  CHECK(cuInit(0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, 0));

  CUdeviceptr ddata = 0, dout16 = 0, doutbf = 0;
  CHECK(cuMemAlloc(&ddata, N * sizeof(uint16_t)));
  CHECK(cuMemAlloc(&dout16, static_cast<size_t>(GX) * sizeof(uint16_t)));
  CHECK(cuMemAlloc(&doutbf, static_cast<size_t>(GX) * sizeof(uint16_t)));
  CHECK(cuMemcpyHtoD(ddata, data.data(), N * sizeof(uint16_t)));

  CUmodule mod = nullptr;
  CHECK(cuModuleLoadData(&mod, kFp16ReducePtx));
  CUfunction fn = nullptr;
  CHECK(cuModuleGetFunction(&fn, mod, "fp16_reduce"));

  const unsigned int n = N;
  void* params[4] = {&ddata, &dout16, &doutbf,
                     const_cast<unsigned int*>(&n)};
  CHECK(cuLaunchKernel(fn, GX, 1, 1, BX, 1, 1, 0, nullptr, params, nullptr));
  CHECK(cuCtxSynchronize());

  CHECK(cuMemcpyDtoH(out_f16.data(), dout16,
                     static_cast<size_t>(GX) * sizeof(uint16_t)));
  CHECK(cuMemcpyDtoH(out_bf16.data(), doutbf,
                     static_cast<size_t>(GX) * sizeof(uint16_t)));

  for (int b = 0; b < GX; ++b) {
    float sum = 0.0f;
    for (int i = b * BX; i < (b + 1) * BX; ++i) sum += static_cast<float>(i % 4 + 1);
    _Float16 want_f16 = static_cast<_Float16>(sum * sum + sum);  // fma(a, a, a)
    uint16_t want_f16_bits;
    std::memcpy(&want_f16_bits, &want_f16, 2);
    const uint16_t want_bf16_bits = f32_to_bf16_rne(sum * 2.0f);
    if (out_f16[static_cast<size_t>(b)] != want_f16_bits) {
      std::fprintf(stderr, "block %d f16: got %u, expected %u\n", b,
                   out_f16[static_cast<size_t>(b)], want_f16_bits);
      return 1;
    }
    if (out_bf16[static_cast<size_t>(b)] != want_bf16_bits) {
      std::fprintf(stderr, "block %d bf16: got %u, expected %u\n", b,
                   out_bf16[static_cast<size_t>(b)], want_bf16_bits);
      return 1;
    }
  }

  std::printf(
      "PASS: driver-API fp16_reduce (f16/bf16 cvt, fma, shared + bar.sync) "
      "on CPU-emulated ROUGE device (%d blocks x %d threads)\n",
      GX, BX);

  CHECK(cuMemFree(ddata));
  CHECK(cuMemFree(dout16));
  CHECK(cuMemFree(doutbf));
  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));
  return 0;
}
