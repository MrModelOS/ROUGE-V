// Каноническое mma-ядро: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.
//
// Host-эталон повторяет lane-формулы PTX ISA 9.7.16.5.8 и k-возрастающую
// FMA-цепочку — наш документированный порядок накопления (спека порядок не
// определяет, поэтому совпадение проверяется именно с ним, побитово).
// A[r][c] = r*16+c, B[r][c] = 1000+r*8+c (как f16), C = 0.

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

// f16 -> f32 (точная копия интерпретатора, независимая реализация для эталона).
static float h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      uint32_t m = man, shift = 0;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++shift;
      }
      m &= 0x3ffu;
      bits = sign | ((113u - shift) << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
  }
  float f = 0;
  std::memcpy(&f, &bits, 4);
  return f;
}

static uint16_t f2h(float f) {
  uint32_t x = 0;
  std::memcpy(&x, &f, 4);
  uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
  uint32_t e = (x >> 23) & 0xffu;
  uint32_t man = x & 0x7fffffu;
  if (e == 0xffu) return man ? (uint16_t)(sign | 0x7e00u) : (uint16_t)(sign | 0x7c00u);
  int exp = (int)e - 127 + 15;
  if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return sign;
    uint32_t m = man | 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t val = m >> shift;
    uint32_t rem = m & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (val & 1u))) ++val;
    return (uint16_t)(sign | val);
  }
  uint32_t val = ((uint32_t)exp << 10) | (man >> 13);
  uint32_t rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (val & 1u))) ++val;
  return (uint16_t)(sign | val);
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  const std::string path = dir + "/mma_tile.ptx";
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
    if (prog->functions[i].name == "mma_tile") fnIndex = (int)i;
  EXPECT_TRUE(fnIndex >= 0);
  if (fnIndex < 0) return 1;

  // Эталон: полный D-тайл 16x8.
  float ref[16][8];
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 8; ++c) {
      // A[r][k] = r*16+k, B[k][c] = 1000+k*8+c — через f16 туда-обратно,
      // как в ядре (значения маленькие, все точные).
      float sum = 0.0f;
      for (int k = 0; k < 16; ++k) {
        float a = h2f(f2h((float)(r * 16 + k)));
        float b = h2f(f2h((float)(1000 + k * 8 + c)));
        sum = std::fma(a, b, sum);
      }
      ref[r][c] = sum;
    }

  float* out = (float*)std::calloc(128, sizeof(float));
  unsigned long long dev = reinterpret_cast<unsigned long long>(out);
  std::vector<uint8_t> blob(8, 0);
  std::memcpy(blob.data(), &dev, 8);
  if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, 32, 1, 1, &err, nullptr)) {
    std::printf("FAIL exec: %s\n", err.c_str());
    return 1;
  }

  // Сверка: lane l хранит out[l*4+s] для (r,c) по формулам спеки.
  int bad = 0;
  for (int lane = 0; lane < 32; ++lane) {
    int g = lane >> 2, t = lane & 3;
    for (int s = 0; s < 4; ++s) {
      int r = (s < 2) ? g : g + 8;
      int c = t * 2 + (s & 1);
      float got = out[lane * 4 + s];
      float want = ref[r][c];
      uint32_t gb = 0, wb = 0;
      std::memcpy(&gb, &got, 4);
      std::memcpy(&wb, &want, 4);
      if (gb != wb) {
        if (bad < 5)
          std::printf("lane=%d s=%d (r=%d,c=%d): got=%g want=%g\n", lane, s, r, c,
                      (double)got, (double)want);
        ++bad;
      }
    }
  }
  EXPECT_TRUE(bad == 0);
  std::free(out);
  if (failures == 0) std::printf("PASS: mma.sync m16n8k16 bit-exact vs reference\n");
  return failures ? 1 : 0;
}
