// AOT end-to-end test for shared memory + bar.sync (block_reduce kernel).
//
// The kernel is translated by ptx2ir and compiled by clang to a native object
// (see tests/CMakeLists.txt). The host driver plays the ROUGE-V runtime:
// per block it allocates a scratchpad (.shared) and a pthread barrier, then
// runs the block's threads concurrently — matching CUDA SIMT semantics so that
// bar.sync is a real synchronization point. Verification: out[b] must equal
// the sum of data[b*BX .. (b+1)*BX).
//
// The scratchpad size and the .shared symbol offsets come from the kernel
// itself via __rouge_block_reduce_query() (emitted by ptx2ir from the PTX), so
// nothing about the layout is hard-coded in the driver.

#include <pthread.h>
#include <rouge_runtime.h>

#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

// ptx2ir signatures:
//   define void @block_reduce(i64, i64, i32, ptr %launch)
//   define i32 @__rouge_block_reduce_query(ptr %info)
extern "C" void block_reduce(uint64_t data, uint64_t out, uint32_t n,
                             const void* launch);
extern "C" int32_t __rouge_block_reduce_query(RougeKernelInfo* info);

struct Launch {
  int32_t tidx, tiy, tiz;
  int32_t ctaidx, ctaidy, ctaidz;
  int32_t ntidx, ntidy, ntidz;
  int32_t nctaidx, nctaidy, nctaidz;
  uint64_t smem;    // offset 48: scratchpad (block .shared)
  void* barrier;    // offset 56: block barrier (bar.sync)
};

struct BlockState {
  pthread_barrier_t barrier;
  void* scratch = nullptr;
};

int main() {
  const uint32_t N = 4096;
  const int32_t BX = 256;     // threads per block
  const int32_t GX = N / BX;  // blocks per grid

  // Kernel metadata: scratchpad size + .shared layout, exported by ptx2ir.
  RougeKernelInfo info{};
  if (!rouge_kernel_info_query(&__rouge_block_reduce_query, &info)) {
    std::printf("AOT block_reduce FAILED: kernel query returned 0\n");
    return 1;
  }
  if (info.smem_size == 0 || info.shared_count != 1) {
    std::printf("AOT block_reduce FAILED: bad kernel info (smem=%u vars=%u)\n",
                info.smem_size, info.shared_count);
    return 1;
  }
  const uint32_t smem_off = rouge_smem_offset(&info, "smem");
  if (smem_off == UINT32_MAX || info.shared_vars[0].elem_bytes != 4 ||
      info.shared_vars[0].count != 256) {
    std::printf("AOT block_reduce FAILED: unexpected .shared layout\n");
    return 1;
  }
  std::printf("kernel info: smem=%u bytes, var '%s' @%u (%u x %uB)\n",
              info.smem_size, info.shared_vars[0].name, smem_off,
              info.shared_vars[0].count, info.shared_vars[0].elem_bytes);

  std::vector<float> data(N);
  std::vector<float> out(static_cast<size_t>(GX), 0.0f);
  for (uint32_t i = 0; i < N; ++i) data[i] = static_cast<float>(i + 1);

  std::vector<BlockState> blocks(static_cast<size_t>(GX));
  for (auto& b : blocks) {
    pthread_barrier_init(&b.barrier, nullptr, BX);
    b.scratch = rouge_smem_alloc(&info);
    if (!b.scratch) {
      std::printf("AOT block_reduce FAILED: scratchpad alloc\n");
      return 1;
    }
  }

  for (int32_t b = 0; b < GX; ++b) {
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(BX));
    for (int32_t t = 0; t < BX; ++t) {
      Launch L{};
      L.tidx = t;
      L.ctaidx = b;
      L.ntidx = BX;
      L.nctaidx = GX;
      L.smem = reinterpret_cast<uint64_t>(blocks[static_cast<size_t>(b)].scratch);
      L.barrier = &blocks[static_cast<size_t>(b)].barrier;
      threads.emplace_back([&, L]() {
        block_reduce(reinterpret_cast<uint64_t>(data.data()),
                     reinterpret_cast<uint64_t>(out.data()), N, &L);
      });
    }
    for (auto& th : threads) th.join();
  }

  for (auto& b : blocks) {
    pthread_barrier_destroy(&b.barrier);
    rouge_smem_free(b.scratch);
  }

  int bad = 0;
  for (int32_t b = 0; b < GX; ++b) {
    float expected = 0.0f;
    for (int32_t i = b * BX; i < (b + 1) * BX; ++i) expected += data[static_cast<size_t>(i)];
    if (out[static_cast<size_t>(b)] != expected) {
      if (bad < 5)
        std::printf("block %d: got %.6f want %.6f\n", b,
                    out[static_cast<size_t>(b)], expected);
      ++bad;
    }
  }

  if (bad) {
    std::printf("AOT block_reduce FAILED: %d bad blocks\n", bad);
    return 1;
  }
  std::printf("AOT block_reduce OK: %d blocks x %d threads (shared + bar.sync)\n",
              GX, BX);
  return 0;
}
