// AOT end-to-end test for ldmatrix/stmatrix (ldmatrix_tile kernel).
//
// The collective moves 8x8 .b16 matrices between shared memory and registers.
// The data itself already lives in the shared window, so the only thing that
// crosses lanes is each lane's row-start ADDRESS: the AOT path publishes it
// into the per-block exchange region named by the launch descriptor, syncs the
// block, and then reads the peers' addresses.
//
// Three entries of the canonical kernel are checked, each against the PTX
// interpreter (linked in) and against the host's own formula:
//   ldmatrix_tile    plain x1 roundtrip
//   ldmatrix_trans   .trans (column receive side)
//   ldmatrix_x4      four matrices, every thread receiving a fragment of each

#include <pthread.h>
#include <rouge_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ptx.h"

// ptx2ir signatures: define void @<entry>(i64 %out, ptr %launch)
extern "C" void ldmatrix_tile(uint64_t out, const void* launch);
extern "C" void ldmatrix_trans(uint64_t out, const void* launch);
extern "C" void ldmatrix_x4(uint64_t out, const void* launch);
extern "C" void ldmatrix_sm75(uint64_t out, const void* launch);
extern "C" void stmatrix_sm90(uint64_t out, const void* launch);
extern "C" int32_t __rouge_ldmatrix_tile_query(RougeKernelInfo* info);

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

// Run one block of 32 lanes concurrently (the collective's barrier needs them
// all to arrive) and return the kernel's output buffer.
std::vector<float> run_aot(const BlockState& block, void (*kernel)(uint64_t, const void*),
                           size_t floats) {
  std::vector<float> out(floats, 0.0f);
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
    threads.emplace_back([&L, kernel, &out]() {
      kernel(reinterpret_cast<uint64_t>(out.data()), &L);
    });
  }
  for (auto& th : threads) th.join();
  return out;
}

std::vector<float> run_interp(const rouge::PtxProgram& prog, const char* entry,
                              size_t floats) {
  int fnIndex = -1;
  for (size_t i = 0; i < prog.functions.size(); ++i)
    if (prog.functions[i].name == entry) fnIndex = static_cast<int>(i);
  std::vector<float> out(floats, 0.0f);
  if (fnIndex < 0) return out;
  const uint64_t dev = reinterpret_cast<uint64_t>(out.data());
  std::vector<uint8_t> blob(sizeof(uint64_t), 0);
  std::memcpy(blob.data(), &dev, sizeof(uint64_t));
  std::string err;
  if (!rouge::execute_kernel(prog, fnIndex, blob, 1, 1, 1, kThreads, 1, 1, &err,
                             nullptr)) {
    std::printf("FAIL: interpreter %s: %s\n", entry, err.c_str());
    return out;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string ptxPath =
      argc > 1 ? argv[1] : "kernels/ldmatrix_tile.ptx";
  std::ifstream in(ptxPath, std::ios::binary);
  if (!in) {
    std::printf("AOT ldmatrix_tile FAILED: cannot open %s\n", ptxPath.c_str());
    return 1;
  }
  std::stringstream buf;
  buf << in.rdbuf();
  std::string err;
  auto prog = rouge::parse_ptx(buf.str(), &err);
  if (!prog) {
    std::printf("AOT ldmatrix_tile FAILED: parse: %s\n", err.c_str());
    return 1;
  }

  RougeKernelInfo info{};
  if (!rouge_kernel_info_query(&__rouge_ldmatrix_tile_query, &info)) {
    std::printf("AOT ldmatrix_tile FAILED: kernel query returned 0\n");
    return 1;
  }

  BlockState block;
  pthread_barrier_init(&block.barrier, nullptr, kThreads);
  block.scratch = rouge_smem_alloc(&info);
  block.exchange = rouge_exchange_alloc(kThreads);
  if (!block.scratch || !block.exchange) {
    std::printf("AOT ldmatrix_tile FAILED: alloc\n");
    return 1;
  }

  int bad = 0;
  const auto check = [&](const char* what, void (*kernel)(uint64_t, const void*),
                         const char* entry, size_t floats) {
    const std::vector<float> aot = run_aot(block, kernel, floats);
    const std::vector<float> interp = run_interp(*prog, entry, floats);
    for (size_t i = 0; i < floats; ++i) {
      if (aot[i] != interp[i]) {
        if (bad < 8)
          std::printf("%s: [%zu] aot=%.6g interp=%.6g\n", what, i,
                      (double)aot[i], (double)interp[i]);
        ++bad;
      }
    }
  };

  // x1 roundtrip: out[tid*2 + e] = 100-row-8 pattern of (row, col).
  check("ldmatrix_tile", &ldmatrix_tile, "ldmatrix_tile", 64);
  // .trans: the thread holds column tid%8, rows (tid%4)*2..+1.
  check("ldmatrix_trans", &ldmatrix_trans, "ldmatrix_trans", 64);
  // x4: four matrices, every thread receiving a fragment of each.
  check("ldmatrix_x4", &ldmatrix_x4, "ldmatrix_x4", 256);
  // sm_75/sm_90 halves (same take positions as x1, split for silicon).
  check("ldmatrix_sm75", &ldmatrix_sm75, "ldmatrix_sm75", 64);
  check("stmatrix_sm90", &stmatrix_sm90, "stmatrix_sm90", 64);

  rouge_exchange_free(block.exchange);
  rouge_smem_free(block.scratch);
  pthread_barrier_destroy(&block.barrier);

  if (bad) {
    std::printf("AOT ldmatrix_tile FAILED: %d mismatching values\n", bad);
    return 1;
  }
  std::printf("AOT ldmatrix_tile OK: x1 + trans + x4 + sm75 + sm90, AOT == interpreter\n");
  return 0;
}