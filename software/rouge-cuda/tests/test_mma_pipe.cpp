// Сквозной тензорный пайплайн: fill shared -> ldmatrix.x4 (A) +
// ldmatrix.x2.trans (B) -> mma.m16n8k16 -> global.
//
// Значения и эталон совпадают с mma_tile.ptx (A[r][c] = r*16+c,
// B[r][c] = 1000+r*8+c, C = 0, k-возрастающая FMA-цепочка), поэтому здесь
// проверяется именно связка коллективов: фрагменты ldmatrix обязаны лечь в
// mma дословно (PTX ISA 9.7.16.5.15/5.8), любое отклонение в ЛЮБОМ из трёх
// мест ломает биты D.

#include "ptx.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

#define EXPECT_TRUE(cond)                                        \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                \
    }                                                            \
  } while (0)

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  const std::string path = dir + "/mma_pipe.ptx";
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::printf("SKIP: no %s\n", path.c_str());
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
    if (prog->functions[i].name == "mma_pipe") fnIndex = (int)i;
  EXPECT_TRUE(fnIndex >= 0);
  if (fnIndex < 0) return 1;

  float* out = (float*)std::calloc(128, sizeof(float));
  unsigned long long dev = reinterpret_cast<unsigned long long>(out);
  std::vector<uint8_t> blob(8, 0);
  std::memcpy(blob.data(), &dev, 8);
  if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, 32, 1, 1, &err,
                             nullptr)) {
    std::printf("FAIL exec: %s\n", err.c_str());
    return 1;
  }

  // Эталон: D[r][c] = sum_k A[r][k]*B[k][c], f32 FMA по k вверх.
  // Все входы точные в f16 (A < 256, B < 2048), округление одно на сложение.
  int bad = 0;
  for (int lane = 0; lane < 32; ++lane) {
    const int g = lane >> 2, t = lane & 3;
    for (int s = 0; s < 4; ++s) {
      const int r = (s < 2) ? g : g + 8;
      const int c = t * 2 + (s & 1);
      float want = 0.0f;
      for (int k = 0; k < 16; ++k)
        want = std::fma(static_cast<float>(r * 16 + k),
                        static_cast<float>(1000 + k * 8 + c), want);
      const float got = out[lane * 4 + s];
      if (got != want) {
        if (bad < 5)
          std::printf("lane=%d s=%d (r=%d,c=%d): got=%g want=%g\n", lane, s, r,
                      c, (double)got, (double)want);
        ++bad;
      }
    }
  }
  EXPECT_TRUE(bad == 0);
  std::free(out);
  if (failures == 0)
    std::printf("PASS: mma pipeline (ldmatrix.x4 + x2.trans + mma) bit-exact\n");
  return failures ? 1 : 0;
}
