/*
 * Узкие форматы Wave 3: e4m3/e5m2/e3m2/e2m3/e2m1 и блочное масштабирование
 * OCP MX.
 *
 * Таблицы и правила — транскрипция интерпретатора (software/rouge-ptx/ptx.cpp,
 * раздел narrow float formats), который в проекте служит источником истины:
 *
 *   OCP OFP8 (Table 1-2): e4m3 — bias 7, max 448 (0x7E), только NaN (0x7F),
 *     без Inf; e5m2 — bias 15, max 57344 (0x7B), Inf 0x7C/0xFC.
 *   OCP MX v1.0 (Table 7): e3m2 — bias 3, max 28; e2m3 — bias 1, max 7.5;
 *     e2m1 — bias 1, max 6; у последних трёх нет ни Inf, ни NaN.
 *   PTX ISA 9.4 (cvt): переходы в эти форматы несут .satfinite (переполнение
 *     насыщается знаковым MAX_NORM), NaN на входе даёт NaN для e4m3/e5m2 и
 *     положительный MAX_NORM для e3m2/e2m3/e2m1.
 *
 * Блочное масштабирование (OCP MX v1.0, разделы 3.1-3.2):
 *   * MXFP8/MXFP4: блок из 32 элементов, общий масштаб E8M0 (степень двойки,
 *     8 бит, без знака); элементы хранятся в e4m3/e5m2 (MXFP8) или e2m1
 *     (MXFP4). Масштаб подбирается так, чтобы максимум блока влез в
 *     representable range элементного формата БЕЗ обрезки (клиппинга).
 *   * NVFP4: блок из 16 элементов e2m1, масштаб блока — E4M3 (без знака,
 *     значения только 2^k), плюс глобальный масштаб FP32 на весь тензор.
 *
 * Все функции — с отказом на некорректных размерах (возвращают 0 или
 * ничего не делают), как и остальное в этой библиотеке.
 */

#include "rouge/numerics.h"

#include <cmath>
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

// Описание узкого формата — те же поля, что в интерпретаторе.
struct NarrowFormat {
  int expBits;
  int mantBits;
  int bias;
  uint32_t maxBits;  // битовый образ MAX_NORM без знака
  uint32_t nanBits;  // канонический NaN без знака; 0 = NaN нет
  int containerBits; // 8 для всех форматов здесь
  int fieldShift;    // смещение поля в контейнере (e2m1: 2)
  int fieldWidth;    // ширина поля (8, 6 или 4)
};

// Константы — дословно из интерпретатора (software/rouge-ptx/ptx.cpp, строки
// kE4M3..kE2M1): maxBits/nanBits заданы в поле формата, а не в контейнере,
// и БЕЗ знакового бита (иначе satfinite дал бы -MAX вместо +MAX — такую
// ошибку нашли в интерпретаторе и починили там и здесь).
constexpr NarrowFormat kE4M3 = {4, 3, 7, 0x7Eu, 0x7Fu, 8, 0, 8};
constexpr NarrowFormat kE5M2 = {5, 2, 15, 0x7Bu, 0x7Fu, 8, 0, 8};
constexpr NarrowFormat kE3M2 = {3, 2, 3, 0x1Fu, 0x00u, 8, 0, 6};
constexpr NarrowFormat kE2M3 = {2, 3, 1, 0x1Fu, 0x00u, 8, 0, 6};
constexpr NarrowFormat kE2M1 = {2, 1, 1, 0x07u, 0x00u, 8, 2, 4};

float narrow_to_f32(uint32_t bits, const NarrowFormat &F) {
  const uint32_t sign = (bits >> (F.fieldWidth - 1)) & 1u;
  const uint32_t exp = (bits >> F.mantBits) & ((1u << F.expBits) - 1u);
  const uint32_t man = bits & ((1u << F.mantBits) - 1u);
  const uint32_t fsign = sign << 31;
  const uint32_t maxExp = (1u << F.expBits) - 1u;
  uint32_t fbits;
  if (exp == 0) {
    if (man == 0) {
      fbits = fsign;
    } else {
      uint32_t m = man, shift = 0;
      while ((m & (1u << F.mantBits)) == 0) {
        m <<= 1;
        ++shift;
      }
      m &= (1u << F.mantBits) - 1u;
      const int fexp = 127 - F.bias + 1 - static_cast<int>(shift);
      fbits = fsign | (static_cast<uint32_t>(fexp) << 23) | (m << (23 - F.mantBits));
    }
  } else if (exp == maxExp) {
    const uint32_t allMant = (1u << F.mantBits) - 1u;
    const bool topNormal = ((F.maxBits >> F.mantBits) & maxExp) == maxExp;
    if (F.nanBits != 0 && (man == allMant || !topNormal)) {
      if (!topNormal && man == 0)
        fbits = fsign | 0x7f800000u; // e5m2 Inf
      else
        fbits = fsign | 0x7fc00000u; // NaN
    } else {
      fbits = fsign |
              (static_cast<uint32_t>(static_cast<int>(exp) - F.bias + 127) << 23) |
              (man << (23 - F.mantBits));
    }
  } else {
    fbits = fsign |
            (static_cast<uint32_t>(static_cast<int>(exp) - F.bias + 127) << 23) |
            (man << (23 - F.mantBits));
  }
  return bit_cast_v<float>(fbits);
}

uint32_t f32_to_narrow(float f, const NarrowFormat &F, bool nanToMax) {
  const uint32_t x = bit_cast_v<uint32_t>(f);
  const uint32_t sign = (x >> 31) & 1u;
  const uint32_t e = (x >> 23) & 0xffu;
  const uint32_t man = x & 0x7fffffu;
  const uint32_t maxExp = (1u << F.expBits) - 1u;
  if (e == 0xffu && man != 0) {
    if (nanToMax) return F.maxBits;
    return (sign << (F.fieldWidth - 1)) | F.nanBits;
  }
  if (e == 0xffu) return (sign << (F.fieldWidth - 1)) | F.maxBits; // Inf -> MAX
  if (e == 0 && man == 0) return sign << (F.fieldWidth - 1);
  const int unbiased = static_cast<int>(e) - 127;
  const int tExp = unbiased + F.bias;
  const int M = F.mantBits;
  if (tExp > static_cast<int>(maxExp))
    return (sign << (F.fieldWidth - 1)) | F.maxBits;
  if (tExp <= 0) {
    if (tExp < -M) return 0;
    const uint32_t full = (1u << 23) | man;
    const int shift = 1 - tExp + (23 - M);
    uint32_t val = full >> shift;
    const uint32_t rem = full & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (val & 1u))) ++val;
    return (sign << (F.fieldWidth - 1)) | (val & ((1u << F.fieldWidth) - 1u));
  }
  uint32_t val = (static_cast<uint32_t>(tExp) << M) | (man >> (23 - M));
  const uint32_t rem = man & ((1u << (23 - M)) - 1u);
  const uint32_t half = 1u << (22 - M);
  if (rem > half || (rem == half && (val & 1u))) {
    ++val;
    if ((val >> M) > maxExp)
      return (sign << (F.fieldWidth - 1)) | F.maxBits;
  }
  return (sign << (F.fieldWidth - 1)) | (val & ((1u << F.fieldWidth) - 1u));
}

// ---- E8M0: масштаб блока как степень двойки (OCP MX) ----------------------
// Код — смещённая степень: значение = 2^(code - 127) (bias 127, как в
// экспоненте float32). Код 255 в MX означает NaN и не используется.
float e8m0_to_f32(uint8_t code) {
  if (code == 255) return std::nanf(""); // NaN в MX
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

uint8_t f32_to_e8m0(float scale) {
  if (!(scale > 0.0f) || !std::isfinite(scale)) return 127; // 1.0 по умолчанию
  int e = 0;
  std::frexp(scale, &e); // scale = m * 2^e, m в [0.5, 1)
  int p = e;             // ceil(log2(scale)) = e: 2^(e-1) < scale <= 2^e
  if (p < -126) p = -126;
  if (p > 127) p = 127;
  return static_cast<uint8_t>(p + 127);
}

}  // namespace

extern "C" {

uint8_t rouge_quantize_e5m2(float f) {
  return static_cast<uint8_t>(f32_to_narrow(f, kE5M2, false));
}
float rouge_dequantize_e5m2(uint8_t u) {
  return narrow_to_f32(u, kE5M2);
}
uint8_t rouge_quantize_e2m1(float f) {
  // f32_to_narrow работает в битах ПОЛЯ формата (4 бита: знак+exp+mantissa),
  // контейнерные биты [5:2] — забота упаковщика, а не квантователя.
  return static_cast<uint8_t>(f32_to_narrow(f, kE2M1, true) & 0x0Fu);
}
float rouge_dequantize_e2m1(uint8_t u) {
  // Поле формата — младшие 4 бита: знак в бите 3.
  return narrow_to_f32(static_cast<uint32_t>(u) & 0x0Fu, kE2M1);
}
uint8_t rouge_quantize_e2m3(float f) {
  // 6-битное поле в младших битах байта.
  return static_cast<uint8_t>(f32_to_narrow(f, kE2M3, true) & 0x3Fu);
}
float rouge_dequantize_e2m3(uint8_t u) {
  return narrow_to_f32(static_cast<uint32_t>(u) & 0x3Fu, kE2M3);
}
uint8_t rouge_quantize_e3m2(float f) {
  return static_cast<uint8_t>(f32_to_narrow(f, kE3M2, true) & 0x3Fu);
}
float rouge_dequantize_e3m2(uint8_t u) {
  return narrow_to_f32(static_cast<uint32_t>(u) & 0x3Fu, kE3M2);
}

// MXFP8: блок из 32 элементов, общий масштаб E8M0.
// Масштаб подбирается так, чтобы максимум блока влез в диапазон e4m3/e5m2
// БЕЗ обрезки; элементы квантуются уже с учётом масштаба.
int rouge_mx_quantize_f32(const float *x, int n, int elem_bytes, uint8_t *out,
                          uint8_t *scale_out) {
  if (!x || !out || !scale_out || n <= 0) return 0;
  if (elem_bytes != 1 && elem_bytes != 2) return 0;
  const NarrowFormat F = (elem_bytes == 1) ? kE4M3 : kE5M2;
  const int block = 32;
  for (int base = 0; base < n; base += block) {
    const int cnt = (n - base < block) ? (n - base) : block;
    // Максимум блока по модулю.
    float amax = 0.0f;
    for (int i = 0; i < cnt; ++i) {
      const float a = std::fabs(x[base + i]);
      if (a > amax) amax = a;
    }
    float scale = 1.0f;
    if (amax > 0.0f) {
      // Максимальный конечный элемент формата.
      const float emax = narrow_to_f32(F.maxBits, F);
      // Минимальная степень двойки, при которой amax/scale <= emax.
      int p = 0;
      while (amax > std::ldexp(emax, p) && p < 127) ++p;
      scale = std::ldexp(1.0f, p);
    }
    const uint8_t sc = f32_to_e8m0(scale);
    *scale_out++ = sc;
    const float inv = 1.0f / e8m0_to_f32(sc);
    for (int i = 0; i < cnt; ++i) {
      const uint32_t q = f32_to_narrow(x[base + i] * inv, F, false);
      out[base + i] = static_cast<uint8_t>(q);
    }
  }
  return (n + block - 1) / block;
}

// MXFP4: блок из 32 элементов e2m1, общий масштаб E8M0. Масштаб тот же, что
// у MXFP8, но элементы уже 4-битные (диапазон меньше).
int rouge_mx_quantize_f32_e2m1(const float *x, int n, uint8_t *out,
                                uint8_t *scale_out) {
  if (!x || !out || !scale_out || n <= 0) return 0;
  const int block = 32;
  for (int base = 0; base < n; base += block) {
    const int cnt = (n - base < block) ? (n - base) : block;
    float amax = 0.0f;
    for (int i = 0; i < cnt; ++i) {
      const float a = std::fabs(x[base + i]);
      if (a > amax) amax = a;
    }
    float scale = 1.0f;
    if (amax > 0.0f) {
      const float emax = narrow_to_f32(kE2M1.maxBits, kE2M1);
      int p = 0;
      while (amax > std::ldexp(emax, p) && p < 127) ++p;
      scale = std::ldexp(1.0f, p);
    }
    const uint8_t sc = f32_to_e8m0(scale);
    *scale_out++ = sc;
    const float inv = 1.0f / e8m0_to_f32(sc);
    for (int i = 0; i < cnt; ++i) {
      const uint32_t q = f32_to_narrow(x[base + i] * inv, kE2M1, true);
      out[base + i] = static_cast<uint8_t>(q & 0x0Fu);
    }
  }
  return (n + block - 1) / block;
}

// NVFP4: блок из 16 элементов e2m1, масштаб блока — E4M3 (знаковый бит 0,
// допустимы только значения 2^k), плюс глобальный масштаб FP32.
// Возвращает число блоков; scale_out получает по одному байту на блок
// (код E4M3 масштаба блока), global_scale — на вход/выход.
int rouge_nvfp4_quantize_f32(const float *x, int n, float global_scale,
                             uint8_t *out, uint8_t *scale_out) {
  if (!x || !out || !scale_out || n <= 0) return 0;
  if (!(global_scale > 0.0f)) global_scale = 1.0f;
  const int block = 16;
  for (int base = 0; base < n; base += block) {
    const int cnt = (n - base < block) ? (n - base) : block;
    float amax = 0.0f;
    for (int i = 0; i < cnt; ++i) {
      const float a = std::fabs(x[base + i]) * global_scale;
      if (a > amax) amax = a;
    }
    // Масштаб блока в E4M3: допустимы только степени двойки, диапазон
    // e4m3 — до 448. Берём наименьшую степень двойки, при которой amax влезает.
    float bscale = 1.0f;
    if (amax > 0.0f) {
      const float emax = narrow_to_f32(kE4M3.maxBits, kE4M3);
      int p = 0;
      while (amax > std::ldexp(emax, p) && p < 127) ++p;
      bscale = std::ldexp(1.0f, p);
    }
    // В E4M3 кодируем значение как есть (оно — степень двойки).
    const uint8_t sc = static_cast<uint8_t>(f32_to_narrow(bscale, kE4M3, false));
    *scale_out++ = sc;
    const float inv = 1.0f / (narrow_to_f32(sc, kE4M3) * global_scale);
    for (int i = 0; i < cnt; ++i) {
      const uint32_t q = f32_to_narrow(x[base + i] * inv, kE2M1, true);
      out[base + i] = static_cast<uint8_t>(q & 0x0Fu);
    }
  }
  return (n + block - 1) / block;
}

// Обратная операция для MXFP8/MXFP4: элементы и масштабы -> f32.
void rouge_mx_dequantize_f32(const uint8_t *in, const uint8_t *scales, int n,
                             int elem_bytes, float *out) {
  if (!in || !scales || !out || n <= 0) return;
  if (elem_bytes != 1 && elem_bytes != 2) return;
  const NarrowFormat F = (elem_bytes == 1) ? kE4M3 : kE5M2;
  const int block = 32;
  for (int base = 0; base < n; base += block) {
    const int cnt = (n - base < block) ? (n - base) : block;
    const float s = e8m0_to_f32(scales[base / block]);
    for (int i = 0; i < cnt; ++i)
      out[base + i] = narrow_to_f32(in[base + i], F) * s;
  }
}

void rouge_mx_dequantize_f32_e2m1(const uint8_t *in, const uint8_t *scales,
                                  int n, float *out) {
  if (!in || !scales || !out || n <= 0) return;
  const int block = 32;
  for (int base = 0; base < n; base += block) {
    const int cnt = (n - base < block) ? (n - base) : block;
    const float s = e8m0_to_f32(scales[base / block]);
    for (int i = 0; i < cnt; ++i)
      out[base + i] =
          narrow_to_f32(static_cast<uint32_t>(in[base + i]) & 0x0Fu, kE2M1) * s;
  }
}

}  // extern "C"
