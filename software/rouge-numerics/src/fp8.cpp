/*
 * FP8 GEMM Wave 2: C[M][N] f32 = A[M][K] e4m3 * B[K][N] e4m3, f32 accumulate.
 *
 * Формат — OCP OFP8 E4M3 (Table 1-2, bias 7, max 448 = 0x7E, NaN только
 * 0x7F/0xFF, без Inf): те же таблицы, что в интерпретаторе PTX
 * (software/rouge-ptx/ptx.cpp, kE4M3) и в AOT-декодере ptx2ir. Кодирование —
 * дословная транскрипция f32_to_narrow() оттуда: RNE, satfinite (переполнение
 * в знаковый MAX_NORM), NaN в знаковый NaN, signed zero сохраняется.
 * Декодирование — транскрипция narrow_to_f32(). Совпадение с интерпретатором
 * проверяется исчерпывающе: все 256 байтовых значений туда-обратно.
 *
 * Само умножение — decode в f32-буферы + обычный блочный rouge_gemm_f32
 * (упаковка панелей, автотюнер, split_k — всё переиспользуется). Декод стоит
 * O(MK+KN) против O(MNK) счёта, для больших K пренебрежимо; отдельный
 * упакованный fp8-микроядро — следующий шаг, не этот файл.
 */

#include "rouge/numerics.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// --- бит-каст без UB (тот же приём, что в интерпретаторе) ---
template <typename To, typename From>
static To bit_cast_v(const From& f) {
  static_assert(sizeof(To) == sizeof(From), "size");
  To t;
  std::memcpy(&t, &f, sizeof(t));
  return t;
}

// OCP OFP8 E4M3: E=4, M=3, bias 7, MAX_NORM 0x7E, NaN 0x7F.
constexpr int kE = 4, kM = 3, kBias = 7;
constexpr uint32_t kMaxBits = 0x7Eu, kNanBits = 0x7Fu;

// Narrow decode: e4m3 байт -> f32. Транскрипция narrow_to_f32().
// e4m3: exp==15 — нормали вплоть до 0x7E, NaN только при мантиссе 0b111.
static float e4m3_to_f32(uint8_t u) {
  const uint32_t sign = (u >> 7) & 1u;
  const uint32_t exp = (u >> 3) & 0xFu;
  const uint32_t man = u & 0x7u;
  const uint32_t fsign = sign << 31;
  uint32_t fbits;
  if (exp == 0) {
    if (man == 0) {
      fbits = fsign;  // +-0
    } else {
      uint32_t m = man;
      uint32_t shift = 0;
      while ((m & 0x8u) == 0) {
        m <<= 1;
        ++shift;
      }
      m &= 0x7u;
      // 127 + (1 - bias) - shift = 121 - shift при bias 7 (ср. 113 для f16).
      fbits = fsign | ((121u - shift) << 23) | (m << 20);
    }
  } else if (exp == 0xFu) {
    if (man == 0x7u)
      fbits = fsign | 0x7fc00000u;  // NaN (пейлоад не хранится: см. encode)
    else
      fbits = fsign | ((15u - 7u + 127u) << 23) | (man << 20);  // ..0x7E
  } else {
    fbits = fsign | ((exp - 7u + 127u) << 23) | (man << 20);
  }
  return bit_cast_v<float>(fbits);
}

// Narrow encode: f32 -> e4m3 байт, RNE + satfinite. Транскрипция
// f32_to_narrow() c roundToZero=false, nanToMax=false.
static uint8_t f32_to_e4m3(float f) {
  const uint32_t x = bit_cast_v<uint32_t>(f);
  const uint32_t sign = (x >> 31) & 1u;
  const uint32_t e = (x >> 23) & 0xffu;
  const uint32_t man = x & 0x7fffffu;
  if (e == 0xffu && man != 0) return (sign << 7) | 0x7Fu;  // NaN
  if (e == 0xffu) return (sign << 7) | 0x7Eu;              // Inf -> MAX
  if (e == 0 && man == 0) return sign << 7;                // signed zero
  const int tExp = static_cast<int>(e) - 127 + 7;
  if (tExp > 15) return (sign << 7) | 0x7Eu;  // overflow -> MAX
  if (tExp <= 0) {
    if (tExp < -3) return 0;  // мимо минимального субнормала (знак теряется: +0)
    const uint32_t full = (1u << 23) | man;
    const int shift = 1 - tExp + 20;
    uint32_t val = full >> shift;
    const uint32_t rem = full & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (val & 1u))) ++val;
    if (val > 0xFFu) val = 0xFFu;
    if ((val >> 3) == 0xFu) return (sign << 7) | 0x7Eu;  // unreachable guard
    return (sign << 7) | (val & 0xFFu);
  }
  uint32_t val = (static_cast<uint32_t>(tExp) << 3) | (man >> 20);
  const uint32_t rem = man & 0xFFFFFu;
  if (rem > 0x80000u || (rem == 0x80000u && (val & 1u))) ++val;
  // Перенос в exp==15: насыщение только на NaN-паттерне (мантисса 0b111) —
  // правило OFP8 "448..464 округляется в 448".
  if ((val >> 3) > 15) return (sign << 7) | 0x7Eu;
  if ((val >> 3) == 15 && (val & 0x7u) == 0x7u)
    return (sign << 7) | 0x7Eu;
  return (sign << 7) | (val & 0xFFu);
}

}  // namespace

extern "C" {

uint8_t rouge_quantize_e4m3(float f) { return f32_to_e4m3(f); }

float rouge_dequantize_e4m3(uint8_t u) { return e4m3_to_f32(u); }

void rouge_gemm_e4m3(int M, int N, int K, const uint8_t *A, const uint8_t *B,
                     float *C, const RougeGemmConfig *cfg) {
  if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C) return;
  std::vector<float> fa((size_t)M * (size_t)K);
  std::vector<float> fb((size_t)K * (size_t)N);
  for (size_t i = 0, n = fa.size(); i < n; ++i) fa[i] = e4m3_to_f32(A[i]);
  for (size_t i = 0, n = fb.size(); i < n; ++i) fb[i] = e4m3_to_f32(B[i]);
  rouge_gemm_f32(M, N, K, fa.data(), fb.data(), C, cfg);
}

}  // extern "C"
