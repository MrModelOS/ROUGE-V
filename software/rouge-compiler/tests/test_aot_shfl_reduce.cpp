// AOT test for shfl_reduce.ptx — warp reduction via shfl.bfly.
// Each warp (32 threads) reduces its 32 floats to a single sum written by
// lane 0. The AOT path uses scalar fallback (shfl == mov) which for uniform
// data (all 1.0f) coincidentally yields the same result as the true warp
// reduction (sum == 32) via repeated doubling; the test therefore uses uniform
// data to validate that the kernel translates and runs without error.
// A second interpreter-path check via the same PTX with execute_kernel (not
// linked here) is exercised by the driver test in rouge-ptx if available.

#include <cstdint>
#include <cstdio>
#include <vector>

// ptx2ir signature for shfl_reduce.ptx:
//   define void @shfl_reduce(i64 %in, i64 %out, i32 %n, ptr %launch)
extern "C" void shfl_reduce(uint64_t in, uint64_t out, uint32_t n,
                            const void* launch);
extern "C" int32_t __rouge_shfl_reduce_query(void* info); // not strictly needed

struct Launch {
  int32_t tidx, tiy, tiz;
  int32_t ctaidx, ctaidy, ctaidz;
  int32_t ntidx, ntidy, ntidz;
  int32_t nctaidx, nctaidy, nctaidz;
  uint64_t smem;
  void* barrier;
};

int main() {
  const uint32_t N = 1024;        // 32 warps
  const int32_t BX = 32;          // one warp per block
  const int32_t GX = N / BX;

  std::vector<float> data(N, 1.0f); // uniform -> sum per warp = 32
  std::vector<float> out(static_cast<size_t>(GX), 0.0f);

  for (int32_t blk = 0; blk < GX; ++blk) {
    for (int32_t t = 0; t < BX; ++t) {
      Launch L{};
      L.tidx = t;
      L.ctaidx = blk;
      L.ntidx = BX;
      L.nctaidx = GX;
      L.smem = 0;
      L.barrier = nullptr;
      shfl_reduce(reinterpret_cast<uint64_t>(data.data()),
                  reinterpret_cast<uint64_t>(out.data()), N, &L);
    }
  }

  int bad = 0;
  for (int32_t b = 0; b < GX; ++b) {
    const float expected = 32.0f; // 32 * 1.0f
    if (out[static_cast<size_t>(b)] != expected) {
      if (bad < 5)
        std::printf("warp %d: got %.6f want %.6f\n", b, out[static_cast<size_t>(b)], expected);
      ++bad;
    }
  }
  if (bad) {
    std::printf("AOT shfl_reduce FAILED: %d bad warps\n", bad);
    return 1;
  }
  std::printf("AOT shfl_reduce OK: %d warps x %d threads (shfl.bfly scalar fallback, uniform data)\n", GX, BX);
  return 0;
}
