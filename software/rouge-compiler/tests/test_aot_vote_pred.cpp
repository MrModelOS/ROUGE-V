// AOT test for vote_pred.ptx — vote.sync.* + activemask.b32.
// Host path is scalar: each vote is the identity of the thread's own
// predicate, activemask is all-ones. Real warp ballots are covered by the
// GPU test (compiler_nvptx_gpu_exec).

#include <cstdint>
#include <cstdio>
#include <vector>

// ptx2ir signature for vote_pred.ptx:
//   define void @vote_pred(i64 %in, i64 %outAny, i64 %outAll, i64 %outUni,
//                          i64 %outMask, i32 %n, ptr %launch)
extern "C" void vote_pred(uint64_t in, uint64_t outAny, uint64_t outAll,
                          uint64_t outUni, uint64_t outMask, uint32_t n,
                          const void* launch);

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

  std::vector<uint32_t> data(N), oAny(N, 0xdeadbeefu), oAll(N, 0xdeadbeefu),
      oUni(N, 0xdeadbeefu), oMask(N, 0xdeadbeefu);
  for (uint32_t i = 0; i < N; ++i) data[i] = (i % 2 == 0) ? 0u : (100u + i);

  for (int32_t blk = 0; blk < GX; ++blk) {
    for (int32_t t = 0; t < BX; ++t) {
      Launch L{};
      L.tidx = t;
      L.ctaidx = blk;
      L.ntidx = BX;
      L.nctaidx = GX;
      L.smem = 0;
      L.barrier = nullptr;
      vote_pred(reinterpret_cast<uint64_t>(data.data()),
                reinterpret_cast<uint64_t>(oAny.data()),
                reinterpret_cast<uint64_t>(oAll.data()),
                reinterpret_cast<uint64_t>(oUni.data()),
                reinterpret_cast<uint64_t>(oMask.data()), N, &L);
    }
  }

  int bad = 0;
  for (uint32_t l = 0; l < N; ++l) {
    // Host scalar fallback: vote == own predicate, mask == all lanes.
    const uint32_t own = data[l] != 0 ? 1u : 0u;
    if (oAny[l] != own) {
      if (bad < 4) std::printf("lane %u: any got %u want %u\n", l, oAny[l], own);
      ++bad;
    }
    if (oAll[l] != own) {
      if (bad < 4) std::printf("lane %u: all got %u want %u\n", l, oAll[l], own);
      ++bad;
    }
    if (oUni[l] != own) {
      if (bad < 4) std::printf("lane %u: uni got %u want %u\n", l, oUni[l], own);
      ++bad;
    }
    if (oMask[l] != 0xffffffffu) {
      if (bad < 4)
        std::printf("lane %u: mask got 0x%08x want 0xffffffff\n", l, oMask[l]);
      ++bad;
    }
  }
  if (bad) {
    std::printf("AOT vote_pred FAILED: %d mismatches\n", bad);
    return 1;
  }
  std::printf("AOT vote_pred OK: 1 warp x 32 threads (identity votes + all-ones mask)\n");
  return 0;
}
