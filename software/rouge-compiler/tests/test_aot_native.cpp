// AOT native end-to-end test: the vadd kernel is translated by ptx2ir into
// LLVM IR and compiled to a native object by clang (in tests/CMakeLists.txt).
// This host program plays the role of the ROUGE-V runtime: it fills the launch
// descriptor, walks the grid (block x thread) and calls the compiled kernel,
// then verifies c[i] == a[i] + b[i].
//
// This is the AOT path: no PTX interpreter, no instruction-by-instruction
// emulation — the kernel runs as native machine code.

#include <cstdint>
#include <cstdio>
#include <vector>

// Signature produced by ptx2ir for vadd.ptx:
//   define void @vadd(i64 %p0, i64 %p1, i64 %p2, i32 %p3, ptr %launch)
extern "C" void vadd(uint64_t a, uint64_t b, uint64_t c, uint32_t n,
                     const int32_t* launch);

struct Launch {
  int32_t tidx, tiy, tiz;
  int32_t ctaidx, ctaidy, ctaidz;
  int32_t ntidx, ntidy, ntidz;
  int32_t nctaidx, nctaidy, nctaidz;
};

int main() {
  const uint32_t N = 4096;
  const int32_t BX = 256;          // threads per block
  const int32_t GX = N / BX;       // blocks per grid

  std::vector<float> a(N), b(N), c(N, 0.0f);
  for (uint32_t i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i % 97) + 0.5f;
    b[i] = static_cast<float>(i % 13) - 0.25f;
  }

  for (int32_t blk = 0; blk < GX; ++blk) {
    for (int32_t t = 0; t < BX; ++t) {
      Launch L{};
      L.tidx = t;
      L.ctaidx = blk;
      L.ntidx = BX;
      L.nctaidx = GX;
      vadd(reinterpret_cast<uint64_t>(a.data()),
           reinterpret_cast<uint64_t>(b.data()),
           reinterpret_cast<uint64_t>(c.data()), N,
           reinterpret_cast<const int32_t*>(&L));
    }
  }

  int bad = 0;
  for (uint32_t i = 0; i < N; ++i) {
    const float expected = a[i] + b[i];
    if (c[i] != expected) {
      if (bad < 5)
        std::printf("mismatch at %u: got %.6f want %.6f\n", i, c[i], expected);
      ++bad;
    }
  }

  if (bad) {
    std::printf("AOT vadd FAILED: %d mismatches\n", bad);
    return 1;
  }
  std::printf("AOT native vadd OK: %u elements, grid=%dx%d block=%d\n", N, GX,
              BX, BX);
  return 0;
}