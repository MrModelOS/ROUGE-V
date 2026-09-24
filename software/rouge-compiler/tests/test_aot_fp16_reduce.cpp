// AOT end-to-end test for the FP16/BF16 path (fp16_reduce kernel).
//
// Exercises: ld/st .f16 and .bf16, .shared f16 tile, bar.sync, cvt.f32.f16,
// cvt.rn.f16.f32, cvt.rn.bf16.f32, cvt.f32.bf16, add.f16, fma.rn.f16,
// add.rn.bf16 — all through ptx2ir's half/bfloat lowering.
//
// The AOT path widens half/bfloat to f32, computes, and narrows once
// (fptrunc, RNE) — bit-identical to the interpreter's software conversions,
// so both paths must produce the very same 16-bit patterns.

#include <pthread.h>
#include <rouge_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

// ptx2ir signatures:
//   define void @fp16_reduce(i64 data, i64 out_f16, i64 out_bf16, i32 n,
//                            ptr %launch)
extern "C" void fp16_reduce(uint64_t data, uint64_t out_f16, uint64_t out_bf16,
                            uint32_t n, const void* launch);
extern "C" int32_t __rouge_fp16_reduce_query(RougeKernelInfo* info);

struct Launch {
  int32_t tidx, tiy, tiz;
  int32_t ctaidx, ctaidy, ctaidz;
  int32_t ntidx, ntidy, ntidz;
  int32_t nctaidx, nctaidy, nctaidz;
  uint64_t smem;
  void* barrier;
};

struct BlockState {
  pthread_barrier_t barrier;
  void* scratch = nullptr;
};

uint16_t f32_to_bf16_rne(float f) {  // same RNE rule as LLVM fptrunc -> bfloat
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if (((x >> 23) & 0xffu) == 0xffu) return static_cast<uint16_t>(x >> 16);
  return static_cast<uint16_t>((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}

int main() {
  const uint32_t N = 1024;
  const int32_t BX = 32;      // f16 tile size; keeps sums exact in bf16
  const int32_t GX = N / BX;

  RougeKernelInfo info{};
  if (!rouge_kernel_info_query(&__rouge_fp16_reduce_query, &info) ||
      info.shared_count != 1 || info.shared_vars[0].name == nullptr ||
      std::strcmp(info.shared_vars[0].name, "sh") != 0 ||
      info.shared_vars[0].elem_bytes != 2 || info.shared_vars[0].count != 32) {
    std::printf("AOT fp16_reduce FAILED: bad kernel info\n");
    return 1;
  }

  // data[i] = 1..4 as binary16 (exactly representable everywhere)
  std::vector<uint16_t> data(N);
  for (uint32_t i = 0; i < N; ++i) {
    const float v = static_cast<float>(i % 4 + 1);
    _Float16 h = static_cast<_Float16>(v);
    std::memcpy(&data[i], &h, 2);
  }
  std::vector<uint16_t> out_f16(static_cast<size_t>(GX), 0);
  std::vector<uint16_t> out_bf16(static_cast<size_t>(GX), 0);

  std::vector<BlockState> blocks(static_cast<size_t>(GX));
  for (auto& b : blocks) {
    pthread_barrier_init(&b.barrier, nullptr, BX);
    b.scratch = rouge_smem_alloc(&info);
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
        fp16_reduce(reinterpret_cast<uint64_t>(data.data()),
                    reinterpret_cast<uint64_t>(out_f16.data()),
                    reinterpret_cast<uint64_t>(out_bf16.data()), N, &L);
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
    float sum = 0.0f;
    for (int32_t i = b * BX; i < (b + 1) * BX; ++i) sum += static_cast<float>(i % 4 + 1);
    // fma.rn.f16 %h5, %h1, %h1, %h1 == acc*acc + acc
    _Float16 want_f16 = static_cast<_Float16>(sum * sum + sum);
    uint16_t want_f16_bits;
    std::memcpy(&want_f16_bits, &want_f16, 2);
    const uint16_t want_bf16_bits = f32_to_bf16_rne(sum * 2.0f);  // add.bf16 x2
    if (out_f16[static_cast<size_t>(b)] != want_f16_bits) {
      std::printf("block %d f16: got %u want %u (sum=%.1f)\n", b,
                  out_f16[static_cast<size_t>(b)], want_f16_bits, sum);
      ++bad;
    }
    if (out_bf16[static_cast<size_t>(b)] != want_bf16_bits) {
      std::printf("block %d bf16: got %u want %u (sum=%.1f)\n", b,
                  out_bf16[static_cast<size_t>(b)], want_bf16_bits, sum);
      ++bad;
    }
  }

  if (bad) {
    std::printf("AOT fp16_reduce FAILED: %d mismatches\n", bad);
    return 1;
  }
  std::printf(
      "AOT fp16_reduce OK: %d blocks x %d threads "
      "(f16/bf16 loads, stores, cvt, fma, shared + bar.sync)\n",
      GX, BX);
  return 0;
}
