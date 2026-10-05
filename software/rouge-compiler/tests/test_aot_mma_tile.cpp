// AOT end-to-end test for mma.sync.aligned (mma_tile kernel, Wave 3 step 2).
//
// The kernel is translated by ptx2ir into LLVM IR and compiled by clang to a
// native object (see tests/CMakeLists.txt). Because a warp collective needs
// data from other lanes and the host has no warp, the AOT path publishes each
// lane's A/B fragments into the per-block exchange region named by the launch
// descriptor, syncs the block, and reads the peers' words.
//
// This driver plays the ROUGE-V runtime: it fills the launch descriptor,
// allocates the block scratchpad, the barrier and the exchange region, then
// runs the block's 32 lanes as concurrent OS threads — the same SIMT schedule
// the interpreter uses, so bar.sync and the collective are real
// synchronisation points.
//
// The reference is the PTX interpreter itself (linked in below): both paths
// must produce identical output bit for bit, and the host then checks the tile
// against the PTX ISA 9.7.16.5.8 lane formulas, so a shared misunderstanding
// cannot pass by agreeing with itself.

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
//   define void @mma_tile(i64 %out, ptr %launch)
//   define i32 @__rouge_mma_tile_query(ptr %info)
extern "C" void mma_tile(uint64_t out, const void* launch);
extern "C" int32_t __rouge_mma_tile_query(RougeKernelInfo* info);

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

// Reference tile: A[r][c] = r*16+c, B[r][c] = 1000 + r*8 + c, C = 0 (mma_tile.ptx).
// D[r][c] = sum_k A[r][k]*B[k][c], computed in f32 with a k-ascending FMA
// chain — the order the interpreter and the AOT lowering both use (the spec
// leaves it unspecified, so agreement is what we verify, not hardware).
float reference_element(int r, int c) {
  float acc = 0.0f;
  for (int k = 0; k < 16; ++k)
    acc = std::fma(static_cast<float>(r * 16 + k),
                   static_cast<float>(1000 + k * 8 + c), acc);
  return acc;
}

// Run the AOT kernel: one block, 32 lanes as concurrent OS threads. They must
// run concurrently: the collective publishes into the exchange region and then
// blocks on the barrier until every lane has published, so running them one at
// a time would deadlock.
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
      mma_tile(reinterpret_cast<uint64_t>(out), &L);
    });
  }
  for (auto& th : threads) th.join();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string ptxPath =
      argc > 1 ? argv[1] : "kernels/mma_tile.ptx";

  // Kernel metadata (scratchpad layout) comes from the PTX itself.
  RougeKernelInfo info{};
  if (!rouge_kernel_info_query(&__rouge_mma_tile_query, &info)) {
    std::printf("AOT mma_tile FAILED: kernel query returned 0\n");
    return 1;
  }

  BlockState block;
  pthread_barrier_init(&block.barrier, nullptr, kThreads);
  block.scratch = rouge_smem_alloc(&info);
  block.exchange = rouge_exchange_alloc(kThreads);
  // mma_tile has no .shared: a null scratchpad is correct there (smem_size 0).
  if ((info.smem_size != 0 && !block.scratch) || !block.exchange) {
    std::printf("AOT mma_tile FAILED: alloc\n");
    return 1;
  }

  float aot[kThreads * 4] = {0};
  run_aot(block, aot);
  rouge_exchange_free(block.exchange);
  rouge_smem_free(block.scratch);
  pthread_barrier_destroy(&block.barrier);

  // Interpreter reference: same kernel, same block, straight from the PTX file.
  std::ifstream in(ptxPath, std::ios::binary);
  if (!in) {
    std::printf("AOT mma_tile FAILED: cannot open %s\n", ptxPath.c_str());
    return 1;
  }
  std::stringstream buf;
  buf << in.rdbuf();
  const std::string text = buf.str();
  std::string err;
  auto prog = rouge::parse_ptx(text, &err);
  if (!prog) {
    std::printf("AOT mma_tile FAILED: parse: %s\n", err.c_str());
    return 1;
  }
  int fnIndex = -1;
  for (size_t i = 0; i < prog->functions.size(); ++i)
    if (prog->functions[i].name == "mma_tile") fnIndex = static_cast<int>(i);
  if (fnIndex < 0) {
    std::printf("AOT mma_tile FAILED: entry mma_tile not found\n");
    return 1;
  }

  float interp[kThreads * 4] = {0};
  {
    const uint64_t dev = reinterpret_cast<uint64_t>(interp);
    std::vector<uint8_t> blob(sizeof(uint64_t), 0);
    std::memcpy(blob.data(), &dev, sizeof(uint64_t));
    if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, kThreads, 1, 1,
                               &err, nullptr)) {
      std::printf("AOT mma_tile FAILED: interpreter: %s\n", err.c_str());
      return 1;
    }
  }

  int bad = 0;
  for (int lane = 0; lane < kThreads; ++lane) {
    const int g = lane >> 2;      // groupID: row
    const int t = lane & 3;       // tigid: column pair
    for (int s = 0; s < 4; ++s) {
      const int r = (s < 2) ? g : g + 8;
      const int c = t * 2 + (s & 1);
      const float want = reference_element(r, c);
      const float gotAot = aot[lane * 4 + s];
      const float gotInterp = interp[lane * 4 + s];
      if (gotAot != want || gotInterp != want) {
        if (bad < 6)
          std::printf("lane %d elem %d (r=%d c=%d): aot=%.6g interp=%.6g "
                      "want=%.6g\n",
                      lane, s, r, c, (double)gotAot, (double)gotInterp,
                      (double)want);
        ++bad;
      }
    }
  }

  if (bad) {
    std::printf("AOT mma_tile FAILED: %d mismatching elements\n", bad);
    return 1;
  }
  std::printf("AOT mma_tile OK: %d elements, AOT == interpreter == reference\n",
              kThreads * 4);
  return 0;
}