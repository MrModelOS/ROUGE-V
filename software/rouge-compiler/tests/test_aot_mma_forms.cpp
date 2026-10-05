// AOT end-to-end test for every supported mma.sync form (mma_forms kernel).
//
// Sixteen entries, two families per form (see the kernel header):
//   *_ones: uniform 1.0 fragments, C=0 -> every D element is exactly K.
//     Checked AOT against the hand bit patterns (decode/accumulate/encode).
//   *_lane: A=BASE|lane, B=BASE|(lane+100), C=0 — every lane publishes distinct
//     fragments, so any peer mix-up in the exchange changes D. Checked AOT
//     against the interpreter bitwise (the interpreter itself is verified by
//     test_mma and test_mma_forms against hand references).
//
// Like test_aot_mma_tile, this driver plays the ROUGE-V runtime: one block of
// 32 lanes as concurrent OS threads, sharing the scratchpad, the barrier and
// the per-block exchange region named by the launch descriptor.

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

extern "C" int32_t __rouge_mma_bf16k16f32_ones_query(RougeKernelInfo* info);

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

typedef void (*KernelFn)(uint64_t, const void*);

extern "C" {
void mma_bf16k16f32_ones(uint64_t, const void*);
void mma_bf16k16f32_lane(uint64_t, const void*);
void mma_e4m3k16f32_ones(uint64_t, const void*);
void mma_e4m3k16f32_lane(uint64_t, const void*);
void mma_e5m2k16f32_ones(uint64_t, const void*);
void mma_e5m2k16f32_lane(uint64_t, const void*);
void mma_u8k16s32_ones(uint64_t, const void*);
void mma_u8k16s32_lane(uint64_t, const void*);
void mma_s8k16s32_ones(uint64_t, const void*);
void mma_s8k16s32_lane(uint64_t, const void*);
void mma_f16k16f16_ones(uint64_t, const void*);
void mma_f16k16f16_lane(uint64_t, const void*);
void mma_f16k8f32_ones(uint64_t, const void*);
void mma_f16k8f32_lane(uint64_t, const void*);
void mma_bf16k8f16_ones(uint64_t, const void*);
void mma_bf16k8f16_lane(uint64_t, const void*);
}

// One entry: kernel fn, interpreter entry name, D-regs per lane, and either
// hand bit patterns (ones) or -1 (lane: compare against the interpreter).
struct Entry {
  const char* name;
  KernelFn kernel;
  int nD;
  int64_t want[4];  // valid when checkHand is true (stored Wider for s32 -1)
  bool checkHand;
};

const Entry kEntries[] = {
    {"mma_bf16k16f32_ones", &mma_bf16k16f32_ones, 4, {0x41800000, 0x41800000, 0x41800000, 0x41800000}, true},
    {"mma_bf16k16f32_lane", &mma_bf16k16f32_lane, 4, {0, 0, 0, 0}, false},
    {"mma_e4m3k16f32_ones", &mma_e4m3k16f32_ones, 4, {0x41800000, 0x41800000, 0x41800000, 0x41800000}, true},
    {"mma_e4m3k16f32_lane", &mma_e4m3k16f32_lane, 4, {0, 0, 0, 0}, false},
    {"mma_e5m2k16f32_ones", &mma_e5m2k16f32_ones, 4, {0x41800000, 0x41800000, 0x41800000, 0x41800000}, true},
    {"mma_e5m2k16f32_lane", &mma_e5m2k16f32_lane, 4, {0, 0, 0, 0}, false},
    {"mma_u8k16s32_ones", &mma_u8k16s32_ones, 4, {16, 16, 16, 16}, true},
    {"mma_u8k16s32_lane", &mma_u8k16s32_lane, 4, {0, 0, 0, 0}, false},
    {"mma_s8k16s32_ones", &mma_s8k16s32_ones, 4, {16, 16, 16, 16}, true},
    {"mma_s8k16s32_lane", &mma_s8k16s32_lane, 4, {0, 0, 0, 0}, false},
    {"mma_f16k16f16_ones", &mma_f16k16f16_ones, 2, {0x4C004C00, 0x4C004C00}, true},
    {"mma_f16k16f16_lane", &mma_f16k16f16_lane, 2, {0, 0}, false},
    {"mma_f16k8f32_ones", &mma_f16k8f32_ones, 4, {0x41000000, 0x41000000, 0x41000000, 0x41000000}, true},
    {"mma_f16k8f32_lane", &mma_f16k8f32_lane, 4, {0, 0, 0, 0}, false},
    {"mma_bf16k8f16_ones", &mma_bf16k8f16_ones, 2, {0x48004800, 0x48004800}, true},
    {"mma_bf16k8f16_lane", &mma_bf16k8f16_lane, 2, {0, 0}, false},
};

std::vector<uint32_t> run_aot(const BlockState& block, KernelFn kernel,
                              int nD) {
  std::vector<uint32_t> out((size_t)kThreads * 4, 0xdeadbeefu);
  std::vector<Launch> launches((size_t)kThreads);
  std::vector<std::thread> threads;
  threads.reserve((size_t)kThreads);
  for (int32_t t = 0; t < kThreads; ++t) {
    Launch& L = launches[(size_t)t];
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
  (void)nD;
  return out;
}

std::vector<uint32_t> run_interp(const rouge::PtxProgram& prog,
                                 const char* entry) {
  std::vector<uint32_t> out((size_t)kThreads * 4, 0);
  int fnIndex = -1;
  for (size_t i = 0; i < prog.functions.size(); ++i)
    if (prog.functions[i].name == entry) fnIndex = (int)i;
  if (fnIndex < 0) return out;
  const uint64_t dev = reinterpret_cast<uint64_t>(out.data());
  std::vector<uint8_t> blob(sizeof(uint64_t), 0);
  std::memcpy(blob.data(), &dev, sizeof(uint64_t));
  std::string err;
  if (!rouge::execute_kernel(prog, fnIndex, blob, 1, 1, 1, kThreads, 1, 1,
                             &err, nullptr))
    std::printf("FAIL: interpreter %s: %s\n", entry, err.c_str());
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string ptxPath =
      argc > 1 ? argv[1] : "kernels/mma_forms.ptx";
  std::ifstream in(ptxPath, std::ios::binary);
  if (!in) {
    std::printf("AOT mma_forms FAILED: cannot open %s\n", ptxPath.c_str());
    return 1;
  }
  std::stringstream buf;
  buf << in.rdbuf();
  std::string err;
  auto prog = rouge::parse_ptx(buf.str(), &err);
  if (!prog) {
    std::printf("AOT mma_forms FAILED: parse: %s\n", err.c_str());
    return 1;
  }

  // The forms kernel has no .shared: query any entry for the layout.
  RougeKernelInfo info{};
  if (!rouge_kernel_info_query(&__rouge_mma_bf16k16f32_ones_query, &info)) {
    std::printf("AOT mma_forms FAILED: kernel query returned 0\n");
    return 1;
  }

  BlockState block;
  pthread_barrier_init(&block.barrier, nullptr, kThreads);
  block.scratch = rouge_smem_alloc(&info);
  block.exchange = rouge_exchange_alloc(kThreads);
  if ((info.smem_size != 0 && !block.scratch) || !block.exchange) {
    std::printf("AOT mma_forms FAILED: alloc\n");
    return 1;
  }

  int bad = 0;
  for (const auto& e : kEntries) {
    const std::vector<uint32_t> aot = run_aot(block, e.kernel, e.nD);
    if (e.checkHand) {
      for (int lane = 0; lane < kThreads; ++lane)
        for (int s = 0; s < e.nD; ++s) {
          const uint32_t got = aot[(size_t)lane * (size_t)e.nD + (size_t)s];
          if (got != (uint32_t)e.want[s]) {
            if (bad < 6)
              std::printf("%s ones lane=%d s=%d: aot=0x%08x want=0x%08x\n",
                          e.name, lane, s, got, (uint32_t)e.want[s]);
            ++bad;
          }
        }
      continue;
    }
    const std::vector<uint32_t> interp = run_interp(*prog, e.name);
    for (int lane = 0; lane < kThreads; ++lane)
      for (int s = 0; s < e.nD; ++s) {
        const uint32_t got = aot[(size_t)lane * (size_t)e.nD + (size_t)s];
        const uint32_t want = interp[(size_t)lane * (size_t)e.nD + (size_t)s];
        if (got != want) {
          if (bad < 6)
            std::printf("%s lane=%d s=%d: aot=0x%08x interp=0x%08x\n", e.name,
                        lane, s, got, want);
          ++bad;
        }
      }
  }

  if (block.scratch) rouge_smem_free(block.scratch);
  rouge_exchange_free(block.exchange);
  pthread_barrier_destroy(&block.barrier);

  if (bad) {
    std::printf("AOT mma_forms FAILED: %d mismatching values\n", bad);
    return 1;
  }
  std::printf("AOT mma_forms OK: 8 ones (hand bits) + 8 lane (vs interpreter)\n");
  return 0;
}
