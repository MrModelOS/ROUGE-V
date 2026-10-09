// GEMM на половинных входах: квантование + GEMM против f32-эталона.
//
// Проверяется:
//  1. точечные значения конверсий (1.0, -1.0, 0.5, signed zero, Inf, NaN);
//  2. roundtrip представимых значений (целые 1..32 через f16/bf16 — точно);
//  3. GEMM f16/bf16 против rouge_gemm_f32 на тех же данных (допуск 1e-2
//     относительно шкалы — ошибка квантования входов, не ядра).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "rouge/numerics.h"

static int failures = 0;

#define EXPECT_TRUE(cond)                                        \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                \
    }                                                            \
  } while (0)

static uint32_t fbits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

int main() {
  // --- 1. точечные значения ---
  EXPECT_TRUE(rouge_dequantize_f16(0x3C00u) == 1.0f);
  EXPECT_TRUE(rouge_dequantize_f16(0xBC00u) == -1.0f);
  EXPECT_TRUE(rouge_dequantize_f16(0x3800u) == 0.5f);
  EXPECT_TRUE(fbits(rouge_dequantize_f16(0x8000u)) == 0x80000000u);  // -0
  EXPECT_TRUE(std::isinf(rouge_dequantize_f16(0x7C00u)));
  EXPECT_TRUE(std::isnan(rouge_dequantize_f16(0x7E00u)));
  EXPECT_TRUE(rouge_quantize_f16(1.0f) == 0x3C00u);
  EXPECT_TRUE(rouge_quantize_f16(-0.0f) == 0x8000u);
  EXPECT_TRUE((rouge_quantize_f16(100000.0f) & 0x7FFFu) == 0x7C00u);  // -> Inf
  EXPECT_TRUE((rouge_quantize_f16(NAN) & 0x7FFFu) == 0x7E00u);

  EXPECT_TRUE(rouge_dequantize_bf16(0x3F80u) == 1.0f);
  EXPECT_TRUE(rouge_dequantize_bf16(0xBF80u) == -1.0f);
  EXPECT_TRUE(fbits(rouge_dequantize_bf16(0x8000u)) == 0x80000000u);
  EXPECT_TRUE(std::isinf(rouge_dequantize_bf16(0x7F80u)));
  EXPECT_TRUE(rouge_quantize_bf16(1.0f) == 0x3F80u);

  // --- 2. roundtrip целых 1..32 (представимы в обоих форматах точно) ---
  for (int i = 1; i <= 32; ++i) {
    const float f = (float)i;
    EXPECT_TRUE(rouge_dequantize_f16(rouge_quantize_f16(f)) == f);
    EXPECT_TRUE(rouge_dequantize_bf16(rouge_quantize_bf16(f)) == f);
  }

  // --- 3. GEMM против f32-эталона ---
  std::mt19937 rng(0xF16u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int M = 32, N = 28, K = 40;
  std::vector<float> A((size_t)M * K), B((size_t)K * N);
  for (auto &v : A) v = dist(rng);
  for (auto &v : B) v = dist(rng);
  std::vector<uint16_t> A16(A.size()), B16(B.size()), Ab(A.size()), Bb(B.size());
  for (size_t i = 0; i < A.size(); ++i) A16[i] = rouge_quantize_f16(A[i]);
  for (size_t i = 0; i < B.size(); ++i) B16[i] = rouge_quantize_f16(B[i]);
  for (size_t i = 0; i < A.size(); ++i) Ab[i] = rouge_quantize_bf16(A[i]);
  for (size_t i = 0; i < B.size(); ++i) Bb[i] = rouge_quantize_bf16(B[i]);

  std::vector<float> ref((size_t)M * N, 0), got((size_t)M * N, 0);
  rouge_gemm_f32(M, N, K, A.data(), B.data(), ref.data(), nullptr);
  double scale = 1.0;
  for (float v : ref)
    if (std::fabs((double)v) > scale) scale = std::fabs((double)v);

  rouge_gemm_f16(M, N, K, A16.data(), B16.data(), got.data(), nullptr);
  double md = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    md = std::max(md, std::fabs((double)got[i] - (double)ref[i]));
  if (md / scale > 1e-2) {
    std::printf("f16 GEMM rel err %g\n", md / scale);
    ++failures;
  }
  rouge_gemm_bf16(M, N, K, Ab.data(), Bb.data(), got.data(), nullptr);
  md = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    md = std::max(md, std::fabs((double)got[i] - (double)ref[i]));
  if (md / scale > 1e-2) {
    std::printf("bf16 GEMM rel err %g\n", md / scale);
    ++failures;
  }

  // NULL/нулевые размеры — без записи и без падения.
  rouge_gemm_f16(0, N, K, A16.data(), B16.data(), got.data(), nullptr);
  rouge_gemm_bf16(M, N, K, nullptr, Bb.data(), got.data(), nullptr);

  if (failures == 0) std::printf("PASS: gemm f16/bf16\n");
  return failures ? 1 : 0;
}
