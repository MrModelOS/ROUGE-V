// AOT end-to-end test for the tiled GEMM kernel (gemm_tile).
//
// Each CTA is 16x16 threads (BX=16, BY=16). The 16x16 tiles of A and B (f16)
// are cooperatively loaded into shA/shB, bar.sync, then each thread computes
// dot(row,col) over K=16 in f16 and writes f32 to C. Block-linear layout:
// blockId = ctaId.y * nCtaId.x + ctaId.x, each block's tiles are contiguous
// (A/B at blockId*512 bytes, C at blockId*1024 bytes).
// Exercises cvta.to.shared, ld/st.shared/global.f16/f32, bar.sync, f16
// arithmetic and conversions.

#include <pthread.h>
#include <rouge_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

// ptx2ir signatures:
//   define void @gemm_tile(i64 A, i64 B, i64 C, ptr %launch)
extern "C" void gemm_tile(uint64_t A, uint64_t B, uint64_t C, const void* launch);
extern "C" int32_t __rouge_gemm_tile_query(RougeKernelInfo* info);

struct Launch {
  int32_t tidx, tiy, tiz;
  int32_t ctaidx, ctaidy, ctaidz;
  int32_t ntidx, ntidy, ntidz;
  int32_t nctaidx, nctaidy, nctaidz;
  uint64_t smem;
  void* barrier;
};

struct BlockState {
  pthread_barrier_t barrier;
  void* scratch = nullptr;
};

int main() {
  const int32_t BX = 16;
  const int32_t BY = 16;
  const int32_t GX = 2;
  const int32_t GY = 2;
  const int32_t K = 16;
  const int32_t TILE = BX * BY; // 256
  const int32_t NUM_BLOCKS = GX * GY;

  RougeKernelInfo info{};
  if (!rouge_kernel_info_query(&__rouge_gemm_tile_query, &info)) {
    std::printf("AOT gemm_tile FAILED: kernel query returned 0\n");
    return 1;
  }
  // Expect two shared vars shA/shB each 256 x 2B = 512 bytes, total 1024
  if (info.shared_count != 2 || info.smem_size == 0) {
    std::printf("AOT gemm_tile FAILED: bad kernel info (smem=%u vars=%u)\n",
                info.smem_size, info.shared_count);
    return 1;
  }
  uint32_t offA = rouge_smem_offset(&info, "shA");
  uint32_t offB = rouge_smem_offset(&info, "shB");
  if (offA == UINT32_MAX || offB == UINT32_MAX) {
    std::printf("AOT gemm_tile FAILED: missing shA/shB offsets (offA=%u offB=%u)\n", offA, offB);
    return 1;
  }
  // Validate sizes
  bool ok = false;
  for (uint32_t i = 0; i < info.shared_count; ++i) {
    if (std::strcmp(info.shared_vars[i].name, "shA") == 0) {
      if (info.shared_vars[i].elem_bytes != 2 || info.shared_vars[i].count != 256) {
        std::printf("AOT gemm_tile FAILED: shA layout %u x %uB\n",
                    info.shared_vars[i].count, info.shared_vars[i].elem_bytes);
        return 1;
      }
      ok = true;
    }
    if (std::strcmp(info.shared_vars[i].name, "shB") == 0) {
      if (info.shared_vars[i].elem_bytes != 2 || info.shared_vars[i].count != 256) {
        std::printf("AOT gemm_tile FAILED: shB layout %u x %uB\n",
                    info.shared_vars[i].count, info.shared_vars[i].elem_bytes);
        return 1;
      }
    }
  }
  if (!ok) {
    std::printf("AOT gemm_tile FAILED: shA not found\n");
    return 1;
  }
  std::printf("kernel info: smem=%u bytes, shA@%u shB@%u\n", info.smem_size, offA, offB);

  const size_t N_A = static_cast<size_t>(NUM_BLOCKS) * TILE;
  const size_t N_C = static_cast<size_t>(NUM_BLOCKS) * TILE;
  std::vector<uint16_t> A(N_A), B(N_A);
  std::vector<float> C(N_C, 0.0f);

  // Fill A,B as f16 values 1..4 (exactly representable)
  for (int32_t blk = 0; blk < NUM_BLOCKS; ++blk) {
    for (int i = 0; i < TILE; ++i) {
      float va = static_cast<float>((i % 4) + 1);
      float vb = static_cast<float>(((i + blk) % 4) + 1); // vary per block slightly
      _Float16 ha = static_cast<_Float16>(va);
      _Float16 hb = static_cast<_Float16>(vb);
      std::memcpy(&A[static_cast<size_t>(blk) * TILE + i], &ha, 2);
      std::memcpy(&B[static_cast<size_t>(blk) * TILE + i], &hb, 2);
    }
  }

  std::vector<BlockState> blocks(static_cast<size_t>(NUM_BLOCKS));
  for (auto& b : blocks) {
    pthread_barrier_init(&b.barrier, nullptr, BX * BY);
    b.scratch = rouge_smem_alloc(&info);
    if (!b.scratch) {
      std::printf("AOT gemm_tile FAILED: scratch alloc\n");
      return 1;
    }
  }

  // Launch blocks sequentially, threads in parallel per block
  for (int32_t by = 0; by < GY; ++by) {
    for (int32_t bx = 0; bx < GX; ++bx) {
      int32_t blk = by * GX + bx;
      std::vector<std::thread> threads;
      threads.reserve(static_cast<size_t>(BX * BY));
      for (int32_t ty = 0; ty < BY; ++ty) {
        for (int32_t tx = 0; tx < BX; ++tx) {
          Launch L{};
          L.tidx = tx;
          L.tiy = ty;
          L.ctaidx = bx;
          L.ctaidy = by;
          L.ntidx = BX;
          L.ntidy = BY;
          L.nctaidx = GX;
          L.nctaidy = GY;
          L.smem = reinterpret_cast<uint64_t>(blocks[static_cast<size_t>(blk)].scratch);
          L.barrier = &blocks[static_cast<size_t>(blk)].barrier;
          threads.emplace_back([&, L]() {
            gemm_tile(reinterpret_cast<uint64_t>(A.data()),
                      reinterpret_cast<uint64_t>(B.data()),
                      reinterpret_cast<uint64_t>(C.data()), &L);
          });
        }
      }
      for (auto& th : threads) th.join();
    }
  }

  for (auto& b : blocks) {
    pthread_barrier_destroy(&b.barrier);
    rouge_smem_free(b.scratch);
  }

  // Reference: per block per thread dot product in f16 (as kernel does)
  int bad = 0;
  const float eps = 1e-2f;
  for (int32_t blk = 0; blk < NUM_BLOCKS; ++blk) {
    for (int32_t row = 0; row < BY; ++row) {
      for (int32_t col = 0; col < BX; ++col) {
        _Float16 acc = 0;
        for (int32_t k = 0; k < K; ++k) {
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
        // also allow epsilon for tiny f32 vs f16 rounding differences
        if (diff > eps && diff > std::fabs(want) * 1e-3f) {
          if (bad < 10) {
            std::printf("block %d [%d,%d] idx %zu: got %.6f want %.6f diff %.6f\n",
                        blk, row, col, idx, got, want, diff);
          }
          ++bad;
        }
      }
    }
  }

  if (bad) {
    std::printf("AOT gemm_tile FAILED: %d mismatches / %zu (eps %.2e)\n", bad, N_C, eps);
    return 1;
  }
  std::printf("AOT gemm_tile OK: %dx%d blocks x %dx%d threads (%zu elements, f16 tiled GEMM + bar.sync)\n",
              GX, GY, BX, BY, N_C);
  return 0;
}
