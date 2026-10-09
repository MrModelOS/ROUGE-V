/*
 * FP16/BF16 GEMM Wave 2: C[M][N] f32 = A[M][K] * B[K][N], входы f16/bf16,
 * накопление в f32. Как и FP8 (fp8.cpp): decode в f32-буферы + тот же
 * блочный rouge_gemm_f32 (автотюнер shared). Декод O(MK+KN) против O(MNK).
 *
 * Конверсии — своя реализация по открытым описаниям форматов (IEEE 754
 * half, BF16 из спеки Google/OCP: 1-8-7, bias 127):
 *   f16: 1-5-10, bias 15; субнормалы, Inf/NaN по IEEE.
 *   bf16: младшие 16 бит f32, округление RNE; NaN/Inf сохраняются.
 * Квантование f32->narrow — RNE; переполнение f16 -> Inf (по IEEE, в отличие
 * от satfinite у FP8); bf16 -> Inf. Декод narrow->f32 точный.
 */

#include "rouge/numerics.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

template <typename To, typename From>
static To bit_cast_v(const From &f) {
  static_assert(sizeof(To) == sizeof(From), "size");
  To t;
  std::memcpy(&t, &f, sizeof(t));
  return t;
}

// --- IEEE f16 -> f32 (точно) ---
static float f16_to_f32(uint16_t u) {
  const uint32_t sign = (u >> 15) & 1u;
  const uint32_t exp = (u >> 10) & 0x1fu;
  const uint32_t man = u & 0x3ffu;
  const uint32_t fsign = sign << 31;
  uint32_t fbits;
  if (exp == 0) {
    if (man == 0) {
      fbits = fsign;  // +-0
    } else {
      // Субнормал: нормализовать.
      uint32_t m = man;
      uint32_t shift = 0;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++shift;
      }
      m &= 0x3ffu;
      // 127 + (1 - 15) - shift = 113 - shift.
      fbits = fsign | ((113u - shift) << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    fbits = fsign | 0x7f800000u | (man << 13);  // Inf / NaN (пейлоад)
  } else {
    fbits = fsign | ((exp - 15u + 127u) << 23) | (man << 13);
  }
  return bit_cast_v<float>(fbits);
}

// --- f32 -> f16, RNE; переполнение -> Inf, NaN -> канонический NaN ---
static uint16_t f32_to_f16(float f) {
  const uint32_t x = bit_cast_v<uint32_t>(f);
  const uint32_t sign = (x >> 31) & 1u;
  const uint32_t e = (x >> 23) & 0xffu;
  const uint32_t man = x & 0x7fffffu;
  const uint32_t hsign = sign << 15;
  if (e == 0xffu) {
    if (man != 0) return hsign | 0x7e00u;  // NaN -> канонический
    return hsign | 0x7c00u;                // Inf
  }
  if (e == 0 && man == 0) return hsign;  // signed zero
  const int tExp = static_cast<int>(e) - 127 + 15;
  if (tExp >= 31) return hsign | 0x7c00u;  // overflow -> Inf
  if (tExp <= 0) {
    // Субнормал или ноль: собрать 11 бит (1 implicit + 10) со сдвигом.
    if (tExp < -10) return hsign;  // мимо минимального субнормала
    const uint32_t full = (1u << 23) | man;
    const int shift = 1 - tExp + 13;  // 14 - tExp
    uint32_t val = full >> shift;
    const uint32_t rem = full & ((shift >= 32) ? 0xffffffffu : ((1u << shift) - 1u));
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (val & 1u))) ++val;
    return hsign | (val & 0x3ffu);
  }
  uint32_t val = (static_cast<uint32_t>(tExp) << 10) | (man >> 13);
  const uint32_t rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (val & 1u))) {
    ++val;
    if ((val >> 10) >= 31) return hsign | 0x7c00u;  // перенос в Inf
  }
  return hsign | (val & 0x7fffu);
}

// --- bf16 -> f32: точный сдвиг ---
static float bf16_to_f32(uint16_t u) {
  const uint32_t fbits = static_cast<uint32_t>(u) << 16;
  return bit_cast_v<float>(fbits);
}

// --- f32 -> bf16, RNE; NaN -> канонический NaN, переполнение -> Inf ---
static uint16_t f32_to_bf16(float f) {
  const uint32_t x = bit_cast_v<uint32_t>(f);
  const uint32_t e = (x >> 23) & 0xffu;
  const uint32_t man = x & 0x7fffffu;
  if (e == 0xffu) {
    if (man != 0) return 0x7fc0u;  // NaN
    return (x >> 16) & 0xffffu;    // Inf со знаком
  }
  // RNE по младшим 16 битам: bias 0x7FFF + младший бит старшей половины.
  const uint32_t bias = 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>((x + bias) >> 16);
}

}  // namespace

extern "C" {

uint16_t rouge_quantize_f16(float f) { return f32_to_f16(f); }
float rouge_dequantize_f16(uint16_t u) { return f16_to_f32(u); }
uint16_t rouge_quantize_bf16(float f) { return f32_to_bf16(f); }
float rouge_dequantize_bf16(uint16_t u) { return bf16_to_f32(u); }

void rouge_gemm_f16(int M, int N, int K, const uint16_t *A, const uint16_t *B,
                    float *C, const RougeGemmConfig *cfg) {
  if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C) return;
  std::vector<float> fa((size_t)M * (size_t)K);
  std::vector<float> fb((size_t)K * (size_t)N);
  for (size_t i = 0, n = fa.size(); i < n; ++i) fa[i] = f16_to_f32(A[i]);
  for (size_t i = 0, n = fb.size(); i < n; ++i) fb[i] = f16_to_f32(B[i]);
  rouge_gemm_f32(M, N, K, fa.data(), fb.data(), C, cfg);
}

void rouge_gemm_bf16(int M, int N, int K, const uint16_t *A, const uint16_t *B,
                     float *C, const RougeGemmConfig *cfg) {
  if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C) return;
  std::vector<float> fa((size_t)M * (size_t)K);
  std::vector<float> fb((size_t)K * (size_t)N);
  for (size_t i = 0, n = fa.size(); i < n; ++i) fa[i] = bf16_to_f32(A[i]);
  for (size_t i = 0, n = fb.size(); i < n; ++i) fb[i] = bf16_to_f32(B[i]);
  rouge_gemm_f32(M, N, K, fa.data(), fb.data(), C, cfg);
}

}  // extern "C"
