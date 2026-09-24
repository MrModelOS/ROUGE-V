// gemm_tile (tiled GEMM 16x16 via shared memory in F16) via the CUDA Driver API
// on the CPU-emulated ROUGE device — the interpreter fallback path.
// Kernel is byte-identical with the rouge-compiler AOT test.

#include <rouge/cuda.h>

#include "gemm_tile_ptx.h"

#include <cmath>
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
  const int BX = 16;
  const int BY = 16;
  const int GX = 2;
  const int GY = 2;
  const int K = 16;
  const int TILE = BX * BY;
  const int NUM_BLOCKS = GX * GY;
  const size_t N_A = static_cast<size_t>(NUM_BLOCKS) * TILE;
  const size_t N_C = static_cast<size_t>(NUM_BLOCKS) * TILE;

  std::vector<uint16_t> A(N_A), B(N_A);
  std::vector<float> C(N_C, 0.0f);
  for (int blk = 0; blk < NUM_BLOCKS; ++blk) {
    for (int i = 0; i < TILE; ++i) {
      float va = static_cast<float>((i % 4) + 1);
      float vb = static_cast<float>(((i + blk) % 4) + 1);
      _Float16 ha = static_cast<_Float16>(va);
      _Float16 hb = static_cast<_Float16>(vb);
      std::memcpy(&A[static_cast<size_t>(blk) * TILE + i], &ha, 2);
      std::memcpy(&B[static_cast<size_t>(blk) * TILE + i], &hb, 2);
    }
  }

  CHECK(cuInit(0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, 0));

  CUdeviceptr dA = 0, dB = 0, dC = 0;
  CHECK(cuMemAlloc(&dA, N_A * sizeof(uint16_t)));
  CHECK(cuMemAlloc(&dB, N_A * sizeof(uint16_t)));
  CHECK(cuMemAlloc(&dC, N_C * sizeof(float)));
  CHECK(cuMemcpyHtoD(dA, A.data(), N_A * sizeof(uint16_t)));
  CHECK(cuMemcpyHtoD(dB, B.data(), N_A * sizeof(uint16_t)));

  CUmodule mod = nullptr;
  CHECK(cuModuleLoadData(&mod, kGemmTilePtx));
  CUfunction fn = nullptr;
  CHECK(cuModuleGetFunction(&fn, mod, "gemm_tile"));

  void* params[3] = {&dA, &dB, &dC};
  CHECK(cuLaunchKernel(fn, GX, GY, 1, BX, BY, 1, 0, nullptr, params, nullptr));
  CHECK(cuCtxSynchronize());

  CHECK(cuMemcpyDtoH(C.data(), dC, N_C * sizeof(float)));

  int bad = 0;
  const float eps = 1e-2f;
  for (int blk = 0; blk < NUM_BLOCKS; ++blk) {
    for (int row = 0; row < BY; ++row) {
      for (int col = 0; col < BX; ++col) {
        _Float16 acc = 0;
        for (int k = 0; k < K; ++k) {
          uint16_t a_bits = A[static_cast<size_t>(blk) * TILE + row * 16 + k];
          uint16_t b_bits = B[static_cast<size_t>(blk) * TILE + k * 16 + col];
          _Float16 ah, bh;
          std::memcpy(&ah, &a_bits, 2);
          std::memcpy(&bh, &b_bits, 2);
          acc = acc + ah * bh;
        }
        float want = static_cast<float>(acc);
        size_t idx = static_cast<size_t>(blk) * TILE + row * 16 + col;
        float got = C[idx];
        float diff = std::fabs(got - want);
        if (diff > eps && diff > std::fabs(want) * 1e-3f) {
          if (bad < 10) {
            std::fprintf(stderr, "block %d [%d,%d] idx %zu: got %.6f want %.6f diff %.6f\n",
                         blk, row, col, idx, got, want, diff);
          }
          ++bad;
        }
      }
    }
  }

  if (bad) {
    std::fprintf(stderr, "driver gemm_tile FAILED: %d mismatches / %zu\n", bad, N_C);
    return 1;
  }

  std::printf(
      "PASS: driver-API gemm_tile (f16 tiled GEMM 16x16 + bar.sync) on "
      "CPU-emulated ROUGE device (%dx%d blocks x %dx%d threads)\n",
      GX, GY, BX, BY);

  CHECK(cuMemFree(dA));
  CHECK(cuMemFree(dB));
  CHECK(cuMemFree(dC));
  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));
  return 0;
}
