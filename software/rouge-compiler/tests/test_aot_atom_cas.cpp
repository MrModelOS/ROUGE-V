// AOT end-to-end test for atom.cas / exch / min / max (atom_cas kernel).
//
// Each thread operates on its private slot (g = ctaid*ntid+tid):
//   cas  : 42 -> 100 (CAS)  old 42, new 100
//   exch : 11 -> 55  (xchg) old 11, new 55
//   min  : 50 min 10 (s32)  old 50, new 10
//   max  : 20 max 90 (u32)  old 20, new 90
// Verifies both the returned old values and the final memory contents.
// The host runs each block's threads as real OS threads, so these are real
// hardware atomics (cmpxchg, xchg, smin/umax).

#include <rouge_runtime.h>

#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

// ptx2ir signatures:
//   define void @atom_cas(i64 cas, i64 exch, i64 min, i64 max,
//                          i64 cas_old, i64 exch_old, i64 min_old, i64 max_old,
//                          i32 n, ptr %launch)
extern "C" void atom_cas(uint64_t cas_data, uint64_t exch_data, uint64_t min_data,
                         uint64_t max_data, uint64_t cas_old, uint64_t exch_old,
                         uint64_t min_old, uint64_t max_old, uint32_t n,
                         const void* launch);
extern "C" int32_t __rouge_atom_cas_query(RougeKernelInfo* info);

struct Launch {
  int32_t tidx, tiy, tiz;
  int32_t ctaidx, ctaidy, ctaidz;
  int32_t ntidx, ntidy, ntidz;
  int32_t nctaidx, nctaidy, nctaidz;
  uint64_t smem;
  void* barrier;
};

int main() {
  const uint32_t N = 4096;
  const int32_t BX = 256;
  const int32_t GX = N / BX;

  RougeKernelInfo info{};
  if (!rouge_kernel_info_query(&__rouge_atom_cas_query, &info) ||
      info.shared_count != 0) {
    std::printf("AOT atom_cas FAILED: bad kernel info\n");
    return 1;
  }

  std::vector<uint32_t> cas_data(N, 42);
  std::vector<uint32_t> exch_data(N, 11);
  std::vector<uint32_t> min_data(N, 50);
  std::vector<uint32_t> max_data(N, 20);
  std::vector<uint32_t> cas_old(N, 0);
  std::vector<uint32_t> exch_old(N, 0);
  std::vector<uint32_t> min_old(N, 0);
  std::vector<uint32_t> max_old(N, 0);

  for (int32_t b = 0; b < GX; ++b) {
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(BX));
    for (int32_t t = 0; t < BX; ++t) {
      Launch L{};
      L.tidx = t;
      L.ctaidx = b;
      L.ntidx = BX;
      L.nctaidx = GX;
      threads.emplace_back([&, L]() {
        atom_cas(reinterpret_cast<uint64_t>(cas_data.data()),
                 reinterpret_cast<uint64_t>(exch_data.data()),
                 reinterpret_cast<uint64_t>(min_data.data()),
                 reinterpret_cast<uint64_t>(max_data.data()),
                 reinterpret_cast<uint64_t>(cas_old.data()),
                 reinterpret_cast<uint64_t>(exch_old.data()),
                 reinterpret_cast<uint64_t>(min_old.data()),
                 reinterpret_cast<uint64_t>(max_old.data()), N, &L);
      });
    }
    for (auto& th : threads) th.join();
  }

  int bad = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (cas_old[i] != 42) { std::printf("cas_old[%u]=%u want 42\n", i, cas_old[i]); ++bad; break; }
  }
  for (uint32_t i = 0; i < N; ++i) {
    if (cas_data[i] != 100) { std::printf("cas_data[%u]=%u want 100\n", i, cas_data[i]); ++bad; break; }
  }
  for (uint32_t i = 0; i < N; ++i) {
    if (exch_old[i] != 11) { std::printf("exch_old[%u]=%u want 11\n", i, exch_old[i]); ++bad; break; }
  }
  for (uint32_t i = 0; i < N; ++i) {
    if (exch_data[i] != 55) { std::printf("exch_data[%u]=%u want 55\n", i, exch_data[i]); ++bad; break; }
  }
  for (uint32_t i = 0; i < N; ++i) {
    if (min_old[i] != 50) { std::printf("min_old[%u]=%u want 50\n", i, min_old[i]); ++bad; break; }
  }
  for (uint32_t i = 0; i < N; ++i) {
    if (min_data[i] != 10) { std::printf("min_data[%u]=%u want 10\n", i, min_data[i]); ++bad; break; }
  }
  for (uint32_t i = 0; i < N; ++i) {
    if (max_old[i] != 20) { std::printf("max_old[%u]=%u want 20\n", i, max_old[i]); ++bad; break; }
  }
  for (uint32_t i = 0; i < N; ++i) {
    if (max_data[i] != 90) { std::printf("max_data[%u]=%u want 90\n", i, max_data[i]); ++bad; break; }
  }

  if (bad) {
    std::printf("AOT atom_cas FAILED: %d mismatches\n", bad);
    return 1;
  }
  std::printf("AOT atom_cas OK: %d blocks x %d threads (cas/exch/min/max as real atomics)\n", GX, BX);
  return 0;
}
