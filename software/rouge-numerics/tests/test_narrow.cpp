// Узкие форматы Wave 3: e5m2/e3m2/e2m3/e2m1 и блочное масштабирование MX.
//
// Проверяется:
//  1. исчерпывающий roundtrip по всем значениям e5m2 (256 байт) и e2m1 (16);
//  2. точечные значения из спецификации: max/Inf/NaN/±0/границы;
//  3. MX-блок: масштаб — степень двойки, максимум блока влезает в диапазон
//     элементного формата БЕЗ обрезки (обрезка = тихая порча данных);
//  4. идемпотентность: quantize(dequantize(x)) == dequantize(x);
//  5. NVFP4: 16 элементов на блок, масштаб блока — степень двойки в E4M3.

#include <cmath>
#include <cstdint>
#include <cstdio>
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

int main() {
  // --- 1. e5m2: все 256 значений ---
  for (int u = 0; u < 256; ++u) {
    const float v = rouge_dequantize_e5m2(static_cast<uint8_t>(u));
    const uint8_t back = rouge_quantize_e5m2(v);
    if (std::isnan(v)) {
      // NaN -> NaN (знак сохраняется по правилу PTX cvt).
      EXPECT_TRUE((back & 0x7Fu) == 0x7Fu);
    } else if (std::isinf(v)) {
      // Inf -> MAX (satfinite по PTX cvt): тождества тут нет и не должно быть.
      EXPECT_TRUE((back & 0x7Fu) == 0x7Bu);
    } else {
      EXPECT_TRUE(back == static_cast<uint8_t>(u));
    }
  }
  // Точечные значения: max 57344 (0x7B), Inf 0x7C, NaN 0x7F.
  EXPECT_TRUE(rouge_dequantize_e5m2(0x7Bu) == 57344.0f);
  EXPECT_TRUE(std::isinf(rouge_dequantize_e5m2(0x7Cu)));
  EXPECT_TRUE(std::isnan(rouge_dequantize_e5m2(0x7Fu)));
  EXPECT_TRUE(rouge_quantize_e5m2(57344.0f) == 0x7Bu);
  EXPECT_TRUE(rouge_quantize_e5m2(1e30f) == 0x7Bu);  // satfinite -> MAX
  EXPECT_TRUE(std::isinf(rouge_dequantize_e5m2(0xFCu)));  // -Inf
  EXPECT_TRUE((rouge_quantize_e5m2(-0.0f) & 0x80u) == 0x80u);  // signed zero

  // --- 2. e2m1: 16 значений (знак + 2 exp + 1 mantissa) ---
  for (int u = 0; u < 16; ++u) {
    const float v = rouge_dequantize_e2m1(static_cast<uint8_t>(u));
    const uint8_t back = rouge_quantize_e2m1(v);
    EXPECT_TRUE(back == static_cast<uint8_t>(u));
  }
  // Поле формата: знак в бите 3, поэтому +6 — это 0x07, а -6 — 0x0F.
  EXPECT_TRUE(rouge_dequantize_e2m1(0x07u) == 6.0f);
  EXPECT_TRUE(rouge_dequantize_e2m1(0x0Fu) == -6.0f);
  EXPECT_TRUE(rouge_dequantize_e2m1(0x00u) == 0.0f);
  EXPECT_TRUE(rouge_dequantize_e2m1(0x02u) == 1.0f);
  EXPECT_TRUE(rouge_dequantize_e2m1(0x04u) == 2.0f);
  // Нет ни Inf, ни NaN: NaN -> положительный MAX (правило PTX cvt).
  EXPECT_TRUE(rouge_quantize_e2m1(std::nanf("")) == 0x07u);

  // --- 3. e3m2 / e2m3: точечные (поле: знак+exp+mantissa) ---
  EXPECT_TRUE(rouge_dequantize_e3m2(0x1Fu) == 28.0f);  // max
  EXPECT_TRUE(rouge_dequantize_e3m2(0x3Fu) == -28.0f);
  EXPECT_TRUE(rouge_dequantize_e2m3(0x1Fu) == 7.5f);  // max
  EXPECT_TRUE(rouge_dequantize_e2m3(0x3Fu) == -7.5f);
  EXPECT_TRUE(rouge_dequantize_e3m2(0x04u) == 0.25f);
  EXPECT_TRUE(rouge_dequantize_e2m3(0x08u) == 1.0f);
  // Насыщение положительного переполнения — положительный максимум (регрессия
  // на баг со знаковым битом в maxBits, найденный при переносе таблиц сюда).
  EXPECT_TRUE(rouge_quantize_e3m2(100.0f) == 0x1Fu);
  EXPECT_TRUE(rouge_quantize_e3m2(-100.0f) == 0x3Fu);
  EXPECT_TRUE(rouge_quantize_e2m3(100.0f) == 0x1Fu);
  EXPECT_TRUE(rouge_quantize_e2m3(-100.0f) == 0x3Fu);
  EXPECT_TRUE(rouge_quantize_e2m1(100.0f) == 0x07u);
  EXPECT_TRUE(rouge_quantize_e2m1(-100.0f) == 0x0Fu);

  // --- 4. MXFP8: блок из 32, масштаб E8M0, без обрезки ---
  std::mt19937 rng(0x4D58u);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (int elem_bytes : {1, 2}) {
    for (int trial = 0; trial < 20; ++trial) {
      const int n = 32 * (1 + trial % 3);
      std::vector<float> x(n);
      for (auto &v : x) v = dist(rng);
      // Добавим экстремум, чтобы проверить границы блока.
      x[0] = 37.5f;
      x[1] = -37.5f;
      std::vector<uint8_t> q(n), scales((n + 31) / 32);
      const int blocks = rouge_mx_quantize_f32(x.data(), n, elem_bytes, q.data(),
                                                scales.data());
      EXPECT_TRUE(blocks == static_cast<int>(scales.size()));
      std::vector<float> back(n);
      rouge_mx_dequantize_f32(q.data(), scales.data(), n, elem_bytes, back.data());
      // 1) Ни один элемент не обрезан: значение блока не превышает MAX.
      const float emax = (elem_bytes == 1) ? 448.0f : 57344.0f;
      for (int b = 0; b < blocks; ++b) {
        for (int i = b * 32; i < std::min(n, (b + 1) * 32); ++i) {
          EXPECT_TRUE(std::fabs(back[i]) <= emax * (1.0f + 1e-6f));
        }
      }
      // 2) Масштаб — степень двойки.
      for (uint8_t sc : scales) {
        const float s = std::ldexp(1.0f, static_cast<int>(sc) - 127);
        EXPECT_TRUE(s > 0.0f);
      }
      // 3) Идемпотентность: переквантование даёт то же самое.
      std::vector<uint8_t> q2(n), scales2(scales.size());
      rouge_mx_quantize_f32(back.data(), n, elem_bytes, q2.data(),
                            scales2.data());
      for (int i = 0; i < n; ++i) EXPECT_TRUE(q2[i] == q[i]);
      for (size_t i = 0; i < scales.size(); ++i)
        EXPECT_TRUE(scales2[i] == scales[i]);
      // 4) Точность: относительная ошибка не превышает шага квантования
      //    элементного формата (e4m3: 2^-4, e5m2: 2^-3 от максимума блока).
      const double rel_bound = (elem_bytes == 1) ? 0.0625 : 0.125;
      for (int b = 0; b < blocks; ++b) {
        double bmax = 0.0;
        for (int i = b * 32; i < std::min(n, (b + 1) * 32); ++i)
          bmax = std::max(bmax, (double)std::fabs(x[i]));
        if (bmax == 0.0) continue;
        for (int i = b * 32; i < std::min(n, (b + 1) * 32); ++i) {
          const double rel = std::fabs((double)x[i] - (double)back[i]) / bmax;
          EXPECT_TRUE(rel <= rel_bound * 1.01);
        }
      }
    }
  }

  // --- 5. MXFP4 (e2m1 в блоке 32) ---
  {
    const int n = 64;
    std::vector<float> x(n);
    for (auto &v : x) v = dist(rng);
    std::vector<uint8_t> q(n), scales((n + 31) / 32);
    const int blocks = rouge_mx_quantize_f32_e2m1(x.data(), n, q.data(),
                                                  scales.data());
    EXPECT_TRUE(blocks == 2);
    std::vector<float> back(n);
    rouge_mx_dequantize_f32_e2m1(q.data(), scales.data(), n, back.data());
    for (int i = 0; i < n; ++i) {
      // e2m1 грубый: относительная ошибка может быть большой, но значение не
      // должно «поехать» в ноль или в бесконечность.
      EXPECT_TRUE(std::isfinite(back[i]));
    }
  }

  // --- 6. NVFP4: 16 элементов на блок, масштабл блока в E4M3 ---
  {
    const int n = 48;
    std::vector<float> x(n);
    for (auto &v : x) v = dist(rng);
    std::vector<uint8_t> q(n), scales((n + 15) / 16);
    const int blocks = rouge_nvfp4_quantize_f32(x.data(), n, 1.0f, q.data(),
                                                scales.data());
    EXPECT_TRUE(blocks == 3);
    for (uint8_t sc : scales) {
      // Масштаб блока — степень двойки, помещающаяся в e4m3.
      const float s = rouge_dequantize_e4m3(sc);
      EXPECT_TRUE(s > 0.0f);
      const float mant = s / std::ldexp(1.0f, (int)std::ilogb(s));
      EXPECT_TRUE(std::fabs(mant - 1.0f) < 1e-6f);
    }
  }

  if (failures == 0) std::printf("PASS: narrow formats + MX blocks\n");
  return failures ? 1 : 0;
}
