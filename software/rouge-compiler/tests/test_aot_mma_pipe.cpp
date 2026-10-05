// AOT end-to-end test for the full tensor pipeline (mma_pipe kernel).

#include <pthread.h>
#include <rouge_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ptx.h"

// ptx2ir signature:
//   define void @mma_pipe(i64 %out, ptr %launch)
//   define i32 @__rouge_mma_pipe_query(ptr %info)
extern "C" void mma_pipe(uint64_t out, const void* launch);
extern "C" int32_t __rouge_mma_pipe_query(RougeKernelInfo* info);

namespace {

struct Launch {
  int32_t tidx, tiy, tiz;
  int32_t ctaidx, ctaidy, ctaidz;
  int32_t ntidx, ntidy, ntidz;
  int32_t nctaidx, nctaidy, nctaidz;
  uint64_t smem;      // offset 48: scratchpad (block .shared)
  void* barrier;      // offset 56: block barrier (bar.sync)
  void* exchange;     // offset 64: per-thread warp-exchange region
};

constexpr int32_t kThreads = 32;  // one warp per block

struct BlockState {
  pthread_barrier_t barrier;
  void* scratch = nullptr;
  void* exchange = nullptr;
};

void run_aot(const BlockState& block, float* out) {
  std::vector<Launch> launches(static_cast<size_t>(kThreads));
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(kThreads));
  for (int32_t t = 0; t < kThreads; ++t) {
    Launch& L = launches[static_cast<size_t>(t)];
    L = Launch{};
    L.tidx = t;
    L.ntidx = kThreads;
    L.nctaidx = 1;
    L.smem = reinterpret_cast<uint64_t>(block.scratch);
    L.barrier = const_cast<pthread_barrier_t*>(&block.barrier);
    L.exchange = block.exchange;
    threads.emplace_back([&L, out]() {
      mma_pipe(reinterpret_cast<uint64_t>(out), &L);
    });
  }
  for (auto& th : threads) th.join();
}

float reference_element(int r, int c) {
  float acc = 0.0f;
  for (int k = 0; k < 16; ++k)
    acc = std::fma(static_cast<float>(r * 16 + k),
                   static_cast<float>(1000 + k * 8 + c), acc);
  return acc;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string ptxPath =
      argc > 1 ? argv[1] : "kernels/mma_pipe.ptx";
  std::ifstream in(ptxPath, std::ios::binary);
  if (!in) {
    std::printf("AOT mma_pipe FAILED: cannot open %s\n", ptxPath.c_str());
    return 1;
  }
  std::stringstream buf;
  buf << in.rdbuf();
  std::string err;
  auto prog = rouge::parse_ptx(buf.str(), &err);
  if (!prog) {
    std::printf("AOT mma_pipe FAILED: parse: %s\n", err.c_str());
    return 1;
  }
  int fnIndex = -1;
  for (size_t i = 0; i < prog->functions.size(); ++i)
    if (prog->functions[i].name == "mma_pipe") fnIndex = (int)i;
  if (fnIndex < 0) {
    std::printf("AOT mma_pipe FAILED: entry mma_pipe not found\n");
    return 1;
  }

  RougeKernelInfo info{};
  if (!rouge_kernel_info_query(&__rouge_mma_pipe_query, &info)) {
    std::printf("AOT mma_pipe FAILED: kernel query returned 0\n");
    return 1;
  }
  if (info.smem_size != 768) {
    std::printf("AOT mma_pipe FAILED: smem=%u, want 768\n", info.smem_size);
    return 1;
  }

  BlockState block;
  pthread_barrier_init(&block.barrier, nullptr, kThreads);
  block.scratch = rouge_smem_alloc(&info);
  block.exchange = rouge_exchange_alloc(kThreads);
  if (!block.scratch || !block.exchange) {
    std::printf("AOT mma_pipe FAILED: alloc\n");
    return 1;
  }

  float aot[kThreads * 4] = {0};
  run_aot(block, aot);
  rouge_exchange_free(block.exchange);
  rouge_smem_free(block.scratch);
  pthread_barrier_destroy(&block.barrier);

  float interp[kThreads * 4] = {0};
  {
    const uint64_t dev = reinterpret_cast<uint64_t>(interp);
    std::vector<uint8_t> blob(sizeof(uint64_t), 0);
    std::memcpy(blob.data(), &dev, sizeof(uint64_t));
    if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, kThreads, 1, 1,
                               &err, nullptr)) {
      std::printf("AOT mma_pipe FAILED: interpreter: %s\n", err.c_str());
      return 1;
    }
  }

  int bad = 0;
  for (int lane = 0; lane < kThreads; ++lane) {
    const int g = lane >> 2, t = lane & 3;
    for (int s = 0; s < 4; ++s) {
      const int r = (s < 2) ? g : g + 8;
      const int c = t * 2 + (s & 1);
      const float want = reference_element(r, c);
      if (aot[lane * 4 + s] != want || interp[lane * 4 + s] != want) {
        if (bad < 6)
          std::printf("lane %d elem %d (r=%d c=%d): aot=%.6g interp=%.6g "
                      "want=%.6g\n",
                      lane, s, r, c, (double)aot[lane * 4 + s],
                      (double)interp[lane * 4 + s], (double)want);
        ++bad;
      }
    }
  }

  if (bad) {
    std::printf("AOT mma_pipe FAILED: %d mismatching elements\n", bad);
    return 1;
  }
  std::printf("AOT mma_pipe OK: 128 elements, AOT == interpreter == reference\n");
  return 0;
}
