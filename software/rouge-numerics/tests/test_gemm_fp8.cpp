// FP8 GEMM Wave 2: квантование e4m3 + блочное умножение.
//
// Проверяются три вещи:
//   1. таблицы: все 256 байтовых значений decode(encode) — неподвижные точки
//      acolo, где значение точно представимо; плюс точечные векторы
//      (1.0 -> 0x38, 448 -> 0x7E, inf -> 0x7E, NaN -> 0x7F, -0.0 -> 0x80).
//   2. точность на целых данных 1..8 (все точные в e4m3, все произведения
//      и суммы точные в f32): бит-в-бит с наивным f32-эталоном.
//   3. поведение на дробных данных: относительная ошибка в пределах
//      точности e4m3 (3 бита мантиссы ~ 6% на элемент).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

int main() {
  // --- 1. точечные векторы кодирования ---
  EXPECT_TRUE(rouge_quantize_e4m3(1.0f) == 0x38);
  EXPECT_TRUE(rouge_quantize_e4m3(-1.0f) == 0xB8);
  EXPECT_TRUE(rouge_quantize_e4m3(448.0f) == 0x7E);
  EXPECT_TRUE(rouge_quantize_e4m3(1000.0f) == 0x7E);   // satfinite
  EXPECT_TRUE(rouge_quantize_e4m3(-1000.0f) == 0xFE);  // знаковый MAX
  EXPECT_TRUE(rouge_quantize_e4m3(1.0f / 0.0f) == 0x7E);
  {
    const float nan = std::nanf("");
    EXPECT_TRUE(rouge_quantize_e4m3(nan) == 0x7F);
    EXPECT_TRUE(rouge_quantize_e4m3(-0.0f) == 0x80);  // знак нуля хранится
    EXPECT_TRUE(fbits(rouge_dequantize_e4m3(0x80)) == 0x80000000u);
  }
  EXPECT_TRUE(rouge_dequantize_e4m3(0x38) == 1.0f);
  EXPECT_TRUE(rouge_dequantize_e4m3(0xB8) == -1.0f);

  // --- 2. все 256 значений: decode(encode(x)) неподвижен там, где x точно ---
  // (точные значения: все нормали вида m*2^e, покрытые тестом побитово).
  {
    int bad = 0;
    for (int b = 0; b < 256; ++b) {
      const float f = rouge_dequantize_e4m3((uint8_t)b);
      if (!std::isfinite(f)) continue;  // NaN: encode даёт канонический 0x7F
      const uint8_t rt = rouge_quantize_e4m3(f);
      // e4m3 симметричен по знаку: -x кодируется как x|0x80.
      const uint8_t want = (uint8_t)b;
      if (rt != want && !(want == 0x7F && rt == 0x7F)) {
        if (bad < 5)
          std::printf("roundtrip 0x%02x -> %g -> 0x%02x\n", b, (double)f, rt);
        ++bad;
      }
    }
    EXPECT_TRUE(bad == 0);
  }

  // --- 3. точный GEMM на целых 1..8 (бит-в-бит с наивным f32) ---
  {
    const int M = 32, N = 32, K = 64;
    std::vector<float> af((size_t)M * K), bf((size_t)K * N);
    std::vector<uint8_t> aq(af.size()), bq(bf.size());
    for (size_t i = 0; i < af.size(); ++i) af[i] = (float)(1 + i % 8);
    for (size_t i = 0; i < bf.size(); ++i) bf[i] = (float)(1 + (i * 3) % 8);
    for (size_t i = 0; i < af.size(); ++i) aq[i] = rouge_quantize_e4m3(af[i]);
    for (size_t i = 0; i < bf.size(); ++i) bq[i] = rouge_quantize_e4m3(bf[i]);
    std::vector<float> got((size_t)M * N, 0), want((size_t)M * N, 0);
    rouge_gemm_e4m3(M, N, K, aq.data(), bq.data(), got.data(), nullptr);
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double s = 0;
        for (int k = 0; k < K; ++k) s += (double)af[(size_t)m * K + k] *
                                        (double)bf[(size_t)k * N + n];
        want[(size_t)m * N + n] = (float)s;
      }
    int bad = 0;
    for (size_t i = 0; i < got.size(); ++i)
      if (fbits(got[i]) != fbits(want[i])) {
        if (bad < 5)
          std::printf("[%zu] got=%g want=%g\n", i, (double)got[i],
                      (double)want[i]);
        ++bad;
      }
    EXPECT_TRUE(bad == 0);
  }

  // --- 4. дробные данные: ошибка в пределах точности формата ---
  {
    const int M = 16, N = 16, K = 32;
    std::vector<float> af((size_t)M * K), bf((size_t)K * N);
    for (size_t i = 0; i < af.size(); ++i)
      af[i] = (float)((int)(i * 2654435761u >> 16) % 2000 - 1000) / 100.0f;
    for (size_t i = 0; i < bf.size(); ++i)
      bf[i] = (float)((int)(i * 40503u >> 8) % 2000 - 1000) / 100.0f;
    std::vector<uint8_t> aq(af.size()), bq(bf.size());
    for (size_t i = 0; i < af.size(); ++i) aq[i] = rouge_quantize_e4m3(af[i]);
    for (size_t i = 0; i < bf.size(); ++i) bq[i] = rouge_quantize_e4m3(bf[i]);
    std::vector<float> got((size_t)M * N, 0);
    rouge_gemm_e4m3(M, N, K, aq.data(), bq.data(), got.data(), nullptr);
    // Метрика — против глобального масштаба ответа, а не поэлементно:
    // отдельные выходы около нуля дали бы сотни процентов относительной
    // ошибки из ничего (артефакт метрики, не счёта).
    double maxAbs = 0, maxAbsErr = 0;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double s = 0;
        for (int k = 0; k < K; ++k) s += (double)af[(size_t)m * K + k] *
                                        (double)bf[(size_t)k * N + n];
        const double g = got[(size_t)m * N + n];
        if (std::fabs(s) > maxAbs) maxAbs = std::fabs(s);
        if (std::fabs(g - s) > maxAbsErr) maxAbsErr = std::fabs(g - s);
      }
    const double rel = maxAbsErr / maxAbs;
    std::printf("fp8 fractional max abs err %.4f at scale %.1f (rel %.4f)\n",
                maxAbsErr, maxAbs, rel);
    EXPECT_TRUE(rel < 0.1);
  }

  if (failures == 0) std::printf("PASS: fp8 quantize + gemm_e4m3\n");
  return failures ? 1 : 0;
}
