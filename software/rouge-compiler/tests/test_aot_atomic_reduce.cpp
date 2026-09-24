// AOT end-to-end test for relaxed atomics (atomic_reduce kernel).
//
// The kernel performs global atomics through ptx2ir's atomicrmw lowering:
//   red.add.f32 / red.add.u32  — accumulation without a return value
//   atom.add.u32                — returning form (claim-and-clear on a
//                                 thread-private slot)
// The host runs every block's threads as real OS threads, so these are genuine
// hardware atomics (lock xadd on x86-64, AMO* on RISC-V) contending on the
// same cache lines. Test data are multiples of 0.25 and the total stays well
// below 2^24, so the f32 result is exact no matter in which order the atomic
// adds complete.

#include <rouge_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

// ptx2ir signatures:
//   define void @atomic_reduce(i64 data, i64 sum, i64 count, i64 flags,
//                              i32 n, ptr %launch)
extern "C" void atomic_reduce(uint64_t data, uint64_t sum, uint64_t count,
                              uint64_t flags, uint32_t n, const void* launch);
extern "C" int32_t __rouge_atomic_reduce_query(RougeKernelInfo* info);

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
  if (!rouge_kernel_info_query(&__rouge_atomic_reduce_query, &info) ||
      info.shared_count != 0) {
    std::printf("AOT atomic_reduce FAILED: bad kernel info\n");
    return 1;
  }

  std::vector<float> data(N);
  float expected_sum = 0.0f;
  for (uint32_t i = 0; i < N; ++i) {
    data[i] = static_cast<float>(i % 8 + 1) * 0.25f;
    expected_sum += data[i];
  }

  float sum = 0.0f;
  uint32_t count = 0;
  std::vector<uint32_t> flags(N, 0);

  for (int32_t b = 0; b < GX; ++b) {
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(BX));
    for (int32_t t = 0; t < BX; ++t) {
      Launch L{};
      L.tidx = t;
      L.ctaidx = b;
      L.ntidx = BX;
      L.nctaidx = GX;
      // L by value: the launch descriptor must outlive the thread on its own.
      threads.emplace_back([&, L]() {
        atomic_reduce(reinterpret_cast<uint64_t>(data.data()),
                      reinterpret_cast<uint64_t>(&sum),
                      reinterpret_cast<uint64_t>(&count),
                      reinterpret_cast<uint64_t>(flags.data()), N, &L);
      });
    }
    for (auto& th : threads) th.join();
  }

  int bad = 0;
  if (sum != expected_sum) {
    std::printf("atomic f32 sum: got %.6f want %.6f\n", sum, expected_sum);
    ++bad;
  }
  if (count != N) {
    std::printf("atomic u32 count: got %u want %u\n", count, N);
    ++bad;
  }
  for (uint32_t i = 0; i < N; ++i) {
    if (flags[i] != 0) {  // every thread must observe old == 0 on its slot
      std::printf("flag[%u] = %u, want 0 (atom.add old value)\n", i, flags[i]);
      ++bad;
      break;
    }
  }

  if (bad) {
    std::printf("AOT atomic_reduce FAILED: %d mismatches\n", bad);
    return 1;
  }
  std::printf(
      "AOT atomic_reduce OK: %d blocks x %d threads, sum=%.1f count=%u "
      "(atom.add.f32/u32 + red.add.f32/u32 as real atomics)\n",
      GX, BX, sum, count);
  return 0;
}
