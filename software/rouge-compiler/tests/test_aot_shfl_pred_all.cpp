// AOT test for shfl_pred_all.ptx — predicated up/bfly/idx + legacy down.
// Host path is scalar: identity values, computed SET-predicates
// (up: lane>=8; bfly/idx: true; legacy down: lane+8<=31).
// Real cross-lane values are covered by the GPU test.

#include <cstdint>
#include <cstdio>
#include <vector>

// ptx2ir signature for shfl_pred_all.ptx.
extern "C" void shfl_pred_all(uint64_t in, uint64_t oUp, uint64_t oUpP,
                              uint64_t oBfly, uint64_t oBflyP, uint64_t oIdx,
                              uint64_t oIdxP, uint64_t oDown, uint64_t oDownP,
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

  std::vector<uint32_t> data(N), oUp(N, 0xdeadbeefu), oUpP(N, 0xdeadbeefu),
      oBfly(N, 0xdeadbeefu), oBflyP(N, 0xdeadbeefu), oIdx(N, 0xdeadbeefu),
      oIdxP(N, 0xdeadbeefu), oDown(N, 0xdeadbeefu), oDownP(N, 0xdeadbeefu);
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
      shfl_pred_all(reinterpret_cast<uint64_t>(data.data()),
                    reinterpret_cast<uint64_t>(oUp.data()),
                    reinterpret_cast<uint64_t>(oUpP.data()),
                    reinterpret_cast<uint64_t>(oBfly.data()),
                    reinterpret_cast<uint64_t>(oBflyP.data()),
                    reinterpret_cast<uint64_t>(oIdx.data()),
                    reinterpret_cast<uint64_t>(oIdxP.data()),
                    reinterpret_cast<uint64_t>(oDown.data()),
                    reinterpret_cast<uint64_t>(oDownP.data()), N, &L);
    }
  }

  int bad = 0;
  const auto check = [&](const char* tag, const std::vector<uint32_t>& got,
                         const std::vector<uint32_t>& want) {
    for (uint32_t l = 0; l < N; ++l)
      if (got[l] != want[l]) {
        if (bad < 8)
          std::printf("lane %u: %s got %u want %u\n", l, tag, got[l], want[l]);
        ++bad;
      }
  };
  std::vector<uint32_t> eUp(N), eUpP(N), eBfly(N), eBflyP(N, 1), eIdx(N),
      eIdxP(N, 1), eDown(N), eDownP(N);
  for (uint32_t l = 0; l < N; ++l) {
    eUp[l] = data[l];  // host scalar identity
    eUpP[l] = (l >= 16) ? 1u : 0u;  // up8/cl8 bound model
    eBfly[l] = data[l];
    eIdx[l] = data[l];
    eDown[l] = data[l];
    eDownP[l] = (l + 8 <= 31) ? 1u : 0u;
  }
  check("up", oUp, eUp);
  check("upP", oUpP, eUpP);
  check("bfly", oBfly, eBfly);
  check("bflyP", oBflyP, eBflyP);
  check("idx", oIdx, eIdx);
  check("idxP", oIdxP, eIdxP);
  check("down", oDown, eDown);
  check("downP", oDownP, eDownP);
  if (bad) {
    std::printf("AOT shfl_pred_all FAILED: %d mismatches\n", bad);
    return 1;
  }
  std::printf("AOT shfl_pred_all OK: 1 warp x 32 threads (identity values + SET-predicates)\n");
  return 0;
}
