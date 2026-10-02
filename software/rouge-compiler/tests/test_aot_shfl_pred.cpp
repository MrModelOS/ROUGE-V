// AOT test for shfl_pred.ptx — predicated shfl.sync.down with SET-predicate.
// Host path is scalar: the value is identity (out[i] == in[i]) while the
// predicate is computed from the lane (tid.x mod 32) per PTX ISA 9.7.10.6:
// p=1 for lanes 0..15 (source lane+16 in range), p=0 for lanes 16..31.
// Real cross-lane values are covered by the GPU test (compiler_nvptx_gpu_exec).

#include <cstdint>
#include <cstdio>
#include <vector>

// ptx2ir signature for shfl_pred.ptx:
//   define void @shfl_pred(i64 %in, i64 %out, i64 %outPred, i32 %n, ptr %launch)
extern "C" void shfl_pred(uint64_t in, uint64_t out, uint64_t outPred,
                          uint32_t n, const void* launch);

struct Launch {
  int32_t tidx, tiy, tiz;
  int32_t ctaidx, ctaidy, ctaidz;
  int32_t ntidx, ntidy, ntidz;
  int32_t nctaidx, nctaidy, nctaidz;
  uint64_t smem;
  void* barrier;
};

int main() {
  const uint32_t N = 32;  // one warp
  const int32_t BX = 32;
  const int32_t GX = 1;

  std::vector<uint32_t> data(N), out(N, 0xdeadbeefu), outPred(N, 0xdeadbeefu);
  for (uint32_t i = 0; i < N; ++i) data[i] = 1000u + i * 17u;

  for (int32_t blk = 0; blk < GX; ++blk) {
    for (int32_t t = 0; t < BX; ++t) {
      Launch L{};
      L.tidx = t;
      L.ctaidx = blk;
      L.ntidx = BX;
      L.nctaidx = GX;
      L.smem = 0;
      L.barrier = nullptr;
      shfl_pred(reinterpret_cast<uint64_t>(data.data()),
                reinterpret_cast<uint64_t>(out.data()),
                reinterpret_cast<uint64_t>(outPred.data()), N, &L);
    }
  }

  int bad = 0;
  for (uint32_t l = 0; l < N; ++l) {
    const uint32_t wantPred = (l + 16 <= 31) ? 1u : 0u;
    if (out[l] != data[l]) {
      if (bad < 4)
        std::printf("lane %u: value got %u want %u (host scalar identity)\n", l,
                    out[l], data[l]);
      ++bad;
    }
    if (outPred[l] != wantPred) {
      if (bad < 8)
        std::printf("lane %u: pred got %u want %u\n", l, outPred[l], wantPred);
      ++bad;
    }
  }
  if (bad) {
    std::printf("AOT shfl_pred FAILED: %d mismatches\n", bad);
    return 1;
  }
  std::printf("AOT shfl_pred OK: 1 warp x 32 threads (identity values + SET-predicates)\n");
  return 0;
}
