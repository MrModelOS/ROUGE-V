// AOT end-to-end test for the signed/unsigned comparison kernel.
//
// The kernel computes, for every element, the results of a signed and an
// unsigned comparison of the same pair. If the lowering confuses the
// signedness (it did: the icmp kind was picked by the condition alone), the
// two paths disagree on any negative operand.
//
// ptx2ir signature: define void @cmp_signed(i64 out, i64 in, i32 n, i32 seed,
//                                            ptr %launch)
extern "C" void cmp_signed(unsigned long long out, unsigned long long in,
                            unsigned int n, unsigned int seed, const void* launch);

#include <pthread.h>
#include <rouge_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "ptx.h"

struct Launch {
  int32_t spec[12] = {0};
  uint64_t smem = 0;
  void* barrier = nullptr;
  uint64_t exchange = 0;
};

int main(int argc, char** argv) {
  const std::string ptx_path =
      argc > 1 ? argv[1] : "kernels/cmp_signed.ptx";
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
    if (prog->functions[i].name == "cmp_signed") fnIndex = (int)i;
  if (fnIndex < 0) {
    std::printf("FAIL: no cmp_signed entry\n");
    return 1;
  }
  const auto& fn = prog->functions[fnIndex];

  const int N = 64;
  const unsigned seed = 12345u;
  std::vector<int> input((size_t)N);
  for (int i = 0; i < N; ++i) {
    // The mix must contain negative values: that is where signed and
    // unsigned comparisons differ.
    input[(size_t)i] = (i % 3 == 0) ? -(i + 1) : (i * 7919);
  }
  std::vector<int> native((size_t)N * 4, 0), interp((size_t)N * 4, 0);

  // ---- native (AOT) ----
  for (int i = 0; i < N; ++i) {
    Launch L{};
    L.spec[0] = i;
    L.spec[3] = 0;
    L.spec[6] = N;
    L.spec[9] = 1;
    cmp_signed((unsigned long long)native.data(),
               (unsigned long long)input.data(), (unsigned int)N, seed, &L);
  }

  // ---- interpreter ----
  std::vector<uint8_t> blob((size_t)fn.paramsBlobSize, 0);
  for (const auto& p : fn.params) {
    if (p.index == 0) {
      const unsigned long long v = (unsigned long long)interp.data();
      std::memcpy(blob.data() + p.offset, &v, 8);
    } else if (p.index == 1) {
      const unsigned long long v = (unsigned long long)input.data();
      std::memcpy(blob.data() + p.offset, &v, 8);
    } else if (p.index == 2) {
      const unsigned v = (unsigned)N;
      std::memcpy(blob.data() + p.offset, &v, 4);
    } else if (p.index == 3) {
      const unsigned v = seed;
      std::memcpy(blob.data() + p.offset, &v, 4);
    }
  }
  if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, (unsigned)N, 1, 1,
                             &err, nullptr)) {
    std::printf("FAIL interp: %s\n", err.c_str());
    return 1;
  }

  // ---- the reference: the signed/unsigned distinction is explicit ----
  int bad = 0, first = -1;
  for (int i = 0; i < N; ++i) {
    const int a = input[(size_t)i];
    const int b = (int)seed;
    const int want[4] = {
        a < b ? 1 : 0,                              // signed
        (unsigned)a < (unsigned)b ? 1 : 0,          // unsigned
        a >= b ? 1 : 0,                             // signed
        (unsigned)a >= (unsigned)b ? 1 : 0,         // unsigned
    };
    for (int k = 0; k < 4; ++k) {
      const size_t idx = (size_t)i * 4 + (size_t)k;
      if (native[idx] != want[k] || interp[idx] != want[k]) {
        if (first < 0) first = (int)idx;
        ++bad;
      }
    }
  }
  if (bad) {
    std::printf("FAIL: %d of %d results wrong, first at %d (aot=%d interp=%d)\n",
                bad, N * 4, first, native[(size_t)first], interp[(size_t)first]);
    return 1;
  }
  std::printf(
      "PASS: signed/unsigned comparisons — AOT and interpreter agree with the "
      "explicit reference (%d checks)\n",
      N * 4);
  return 0;
}
