// AOT end-to-end test for the INT8 GEMM kernel (the prmt + dp4a path).
//
// Project invariant: AOT-compiled code and the interpreter agree bit-for-bit.
// The driver calls the NATIVE kernel and the INTERPRETER on identical inputs
// and compares the int32 results. The arithmetic is integer, so there is no
// tolerance: any difference is a bug in the lowering of prmt / dp4a or in the
// shared-memory addressing around them.
//
// The same kernel is executed on a real NVIDIA card by the gpu_exec test
// (compiler_nvptx_cuda_gpu_exec), where it is checked against the host
// reference; the three paths must agree.
//
// ptx2ir signatures:
//   define void @int8_gemm_kernel(i64 A, i64 B, i64 C, i32 M, i32 N, i32 K,
//                                  ptr %launch)
//   define i32 @__rouge_int8_gemm_kernel_query(RougeKernelInfo* info)
#include <pthread.h>
#include <rouge_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "ptx.h"

extern "C" void int8_gemm_kernel(uint64_t A, uint64_t B, uint64_t C, uint32_t M,
                                 uint32_t N, uint32_t K, const void* launch);
extern "C" int32_t __rouge_int8_gemm_kernel_query(RougeKernelInfo* info);

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

int main(int argc, char** argv) {
  const std::string ptx_path =
      argc > 1 ? argv[1] : "kernels/cuda/int8_gemm.ptx";
  FILE* f = std::fopen(ptx_path.c_str(), "rb");
  if (!f) {
    std::printf("SKIP: no %s\n", ptx_path.c_str());
    return 0;
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string text((size_t)n, '\0');
  if (std::fread(&text[0], 1, text.size(), f) != text.size()) {
    std::printf("FAIL: short read\n");
    return 1;
  }
  std::fclose(f);

  std::string err;
  auto prog = rouge::parse_ptx(text, &err);
  if (!prog) {
    std::printf("FAIL parse: %s\n", err.c_str());
    return 1;
  }
  int fnIndex = -1;
  for (size_t i = 0; i < prog->functions.size(); ++i)
    if (prog->functions[i].name == "int8_gemm_kernel") fnIndex = (int)i;
  if (fnIndex < 0) {
    std::printf("FAIL: no int8_gemm_kernel entry\n");
    return 1;
  }
  const auto& fn = prog->functions[fnIndex];

  // Sizes deliberately not multiples of the 16x16 tile.
  const int M = 61, N = 37, K = 83;
  const int BX = 16, BY = 16;
  const int GX = (N + BX - 1) / BX, GY = (M + BY - 1) / BY;

  std::vector<signed char> A((size_t)M * K), B((size_t)K * N);
  std::vector<int> ref((size_t)M * N, 0), native((size_t)M * N, 0);
  uint32_t st = 0x1234567u;
  for (size_t i = 0; i < A.size(); ++i) {
    st = st * 1664525u + 1013904223u;
    A[i] = (signed char)(st >> 13);
  }
  for (size_t i = 0; i < B.size(); ++i) {
    st = st * 1664525u + 1013904223u;
    B[i] = (signed char)(st >> 13);
  }
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (int)A[(size_t)m * K + k] * (int)B[(size_t)k * N + n];
      ref[(size_t)m * N + n] = acc;
    }

  // ---- native (AOT) run: blocks sequential, threads parallel per block ----
  RougeKernelInfo info{};
  if (!rouge_kernel_info_query(&__rouge_int8_gemm_kernel_query, &info) ||
      info.smem_size == 0) {
    std::printf("AOT int8_gemm FAILED: kernel query returned 0\n");
    return 1;
  }
  std::vector<BlockState> blocks((size_t)(GX * GY));
  for (auto& b : blocks) {
    pthread_barrier_init(&b.barrier, nullptr, BX * BY);
    b.scratch = rouge_smem_alloc(&info);
    if (!b.scratch) {
      std::printf("AOT int8_gemm FAILED: scratch alloc\n");
      return 1;
    }
  }
  for (int by = 0; by < GY; ++by) {
    for (int bx = 0; bx < GX; ++bx) {
      const int blk = by * GX + bx;
      std::vector<std::thread> threads;
      threads.reserve((size_t)(BX * BY));
      for (int ty = 0; ty < BY; ++ty) {
        for (int tx = 0; tx < BX; ++tx) {
          Launch L{};
          L.tidx = tx;
          L.tiy = ty;
          L.ctaidx = bx;
          L.ctaidy = by;
          L.ntidx = BX;
          L.ntidy = BY;
          L.nctaidx = GX;
          L.nctaidy = GY;
          L.smem = reinterpret_cast<uint64_t>(blocks[(size_t)blk].scratch);
          L.barrier = &blocks[(size_t)blk].barrier;
          threads.emplace_back([&, L]() {
            int8_gemm_kernel((uint64_t)A.data(), (uint64_t)B.data(),
                             (uint64_t)native.data(), (uint32_t)M, (uint32_t)N,
                             (uint32_t)K, &L);
          });
        }
      }
      for (auto& th : threads) th.join();
    }
  }
  for (auto& b : blocks) {
    pthread_barrier_destroy(&b.barrier);
    rouge_smem_free(b.scratch);
  }

  // ---- interpreter run on the same inputs ----
  // The interpreter writes into its own buffer: the native result must stay
  // intact so that BOTH can be compared with the reference (and with each
  // other).
  std::vector<int> interp((size_t)M * N, 0);
  std::vector<uint8_t> blob((size_t)fn.paramsBlobSize, 0);
  for (const auto& p : fn.params) {
    switch (p.index) {
      case 0: {
        const uint64_t v = (uint64_t)A.data();
        std::memcpy(blob.data() + p.offset, &v, 8);
        break;
      }
      case 1: {
        const uint64_t v = (uint64_t)B.data();
        std::memcpy(blob.data() + p.offset, &v, 8);
        break;
      }
      case 2: {
        const uint64_t v = (uint64_t)interp.data();
        std::memcpy(blob.data() + p.offset, &v, 8);
        break;
      }
      case 3: {
        const uint32_t v = (uint32_t)M;
        std::memcpy(blob.data() + p.offset, &v, 4);
        break;
      }
      case 4: {
        const uint32_t v = (uint32_t)N;
        std::memcpy(blob.data() + p.offset, &v, 4);
        break;
      }
      case 5: {
        const uint32_t v = (uint32_t)K;
        std::memcpy(blob.data() + p.offset, &v, 4);
        break;
      }
      default: break;
    }
  }
  std::memset(interp.data(), 0, interp.size() * sizeof(int));
  if (!rouge::execute_kernel(*prog, fnIndex, blob, (unsigned)GX, (unsigned)GY, 1,
                             (unsigned)BX, (unsigned)BY, 1, &err, nullptr)) {
    std::printf("FAIL interp: %s\n", err.c_str());
    return 1;
  }

  // Both the AOT and the interpreter must equal the reference exactly.
  int bad = 0, first = -1;
  const char* who = "";
  for (size_t i = 0; i < ref.size(); ++i)
    if (native[i] != ref[i]) {
      if (first < 0) {
        first = (int)i;
        who = "AOT";
      }
      ++bad;
    }
  for (size_t i = 0; i < ref.size(); ++i)
    if (interp[i] != ref[i]) {
      if (first < 0) {
        first = (int)i;
        who = "interpreter";
      }
      ++bad;
    }
  if (bad) {
    std::printf("FAIL: %d of %d elements differ (%s first), first at (m=%d,n=%d): "
                "aot=%d interp=%d ref=%d\n",
                bad, M * N, who, first / N, first % N, native[first],
                interp[first], ref[first]);
    return 1;
  }
  std::printf(
      "PASS: AOT int8 GEMM %dx%dx%d bit-identical to the interpreter and to the "
      "host reference (prmt + dp4a path)\n",
      M, N, K);
  return 0;
}
