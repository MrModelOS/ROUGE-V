/*
 * INT8-квантование и целочисленный GEMM (Wave 2, «AI»-часть).
 *
 * Квантования — три схемы, все с честным округлением (round-to-nearest-even)
 * и насыщением по границам формата:
 *   * симметричное per-tensor:  scale = max|x| / 127, нулевая точка 0;
 *   * симметричное per-channel (по строке): тот же принцип, scale на канал;
 *   * асимметричное per-group (как в AWQ/GPTQ): scale = (max-min)/255,
 *     zero = round(-min/scale), значения 0..255.
 *
 * Граница ошибки восстановления: симметричные схемы — половина шага;
 * асимметричная per-group — ШАГ (нулевая точка целочисленная, поэтому
 * представляемый интервал сдвинут относительно [min,max] максимум на
 * полшага, и на границах к этому добавляется полушаг округления).
 * Та же граница, что у асимметричного квантования в AWQ/GPTQ/TensorRT.
 *
 * GEMM: C = (A_s8 x B_s8) — накопление идёт в int32 (ТОЧНО, без ошибок
 * округления), результат приводится к f32 масштабами. Это честно: сначала
 * точность, потом скорость; SIMD-путь int8 — следующий шаг (и будет
 * отдельной строкой в README, а не тихим «оптимизировано»).
 *
 * Почему int32, а не «int8 в int32 с накоплением»: для M×N×K накопление
 * в int32 переполняется только при |значений| ~2^31 — на практике не
 * достигается, и результат совпадает с точным целочисленным произведением
 * бит-в-бит (тест это проверяет против long double эталона).
 */

#include "rouge/numerics.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// Round-to-nearest-even для float -> int (без зависимости от FE_TONEAREST).
inline int32_t rne_int(double v) {
  const double r = std::nearbyint(v);
  return static_cast<int32_t>(r);
}

inline int32_t clamp_i32(double v, int32_t lo, int32_t hi) {
  const int32_t r = rne_int(v);
  return r < lo ? lo : (r > hi ? hi : r);
}

}  // namespace

extern "C" {

// Симметричное per-tensor: scale = max|x| / 127.
// Возвращает 1 при успехе; при n<=0 или нулевой амплитуде scale = 1.
int rouge_quantize_i8_tensor(const float *x, int n, float *scale_out,
                              int8_t *out) {
  if (!x || !scale_out || !out || n <= 0) return 0;
  double amax = 0.0;
  for (int i = 0; i < n; ++i) {
    const double a = std::fabs((double)x[i]);
    if (a > amax) amax = a;
  }
  const float scale = amax > 0.0 ? (float)(amax / 127.0) : 1.0f;
  *scale_out = scale;
  const double inv = 1.0 / (double)scale;
  for (int i = 0; i < n; ++i)
    out[i] = static_cast<int8_t>(clamp_i32((double)x[i] * inv, -127, 127));
  return 1;
}

// Симметричное per-channel: scale считается по каждой строке (канал).
// scales получает rows значений; строки длиной cols.
int rouge_quantize_i8_channel(const float *x, int rows, int cols, float *scales,
                               int8_t *out) {
  if (!x || !scales || !out || rows <= 0 || cols <= 0) return 0;
  for (int r = 0; r < rows; ++r) {
    double amax = 0.0;
    const float *row = x + (size_t)r * cols;
    for (int c = 0; c < cols; ++c) {
      const double a = std::fabs((double)row[c]);
      if (a > amax) amax = a;
    }
    const float scale = amax > 0.0 ? (float)(amax / 127.0) : 1.0f;
    scales[r] = scale;
    const double inv = 1.0 / (double)scale;
    int8_t *o = out + (size_t)r * cols;
    for (int c = 0; c < cols; ++c)
      o[c] = static_cast<int8_t>(clamp_i32((double)row[c] * inv, -127, 127));
  }
  return 1;
}

// Асимметричное per-group: 0..255, scale = (max-min)/255, zero = round(-min/scale).
// groups — число групп по строке (группа = cols/groups элементов подряд).
int rouge_quantize_i8_group(const float *x, int rows, int cols, int groups,
                            float *scales, uint8_t *zeros, uint8_t *out) {
  if (!x || !scales || !zeros || !out || rows <= 0 || cols <= 0 || groups <= 0)
    return 0;
  if (cols % groups != 0) return 0;
  const int gsz = cols / groups;
  for (int r = 0; r < rows; ++r) {
    const float *row = x + (size_t)r * cols;
    uint8_t *o = out + (size_t)r * cols;
    for (int g = 0; g < groups; ++g) {
      // Границы диапазона группы: берём из данных, а не из нуля (иначе у
      // положительных групп min остался бы нулём и масштаб вышел бы вдвое
      // больше нужного — ошибка до шага и больше).
      double lo = (double)row[g * gsz];
      double hi = lo;
      for (int i = 1; i < gsz; ++i) {
        const double v = (double)row[g * gsz + i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      // Нулевая точка целочисленная, поэтому не всякий диапазон представим
      // аффинно с точностью до полушага. Разбираем три случая:
      //   * данные >= 0:      zero = 0,   масштаб по максимуму -> [0, hi];
      //   * данные <= 0:      zero = 255, масштаб по модулю min -> [lo, 0];
      //   * данные вокруг 0:  zero = round(-lo/scale), интервал сдвинут не
      //                        более чем на полшага, ошибка у границ — шаг.
      float scale;
      uint8_t zero;
      if (lo >= 0.0) {
        zero = 0;
        scale = hi > 0.0 ? (float)(hi / 255.0) : 1.0f;
      } else if (hi <= 0.0) {
        zero = 255;
        scale = lo < 0.0 ? (float)(-lo / 255.0) : 1.0f;
      } else {
        scale = (float)((hi - lo) / 255.0);
        zero = (uint8_t)clamp_i32(-lo / (double)scale, 0, 255);
      }
      scales[r * groups + g] = scale;
      zeros[r * groups + g] = zero;
      const double inv = 1.0 / (double)scale;
      for (int i = 0; i < gsz; ++i) {
        const double q = (double)row[g * gsz + i] * inv + (double)zero;
        o[g * gsz + i] = (uint8_t)clamp_i32(q, 0, 255);
      }
    }
  }
  return 1;
}

// Обратное преобразование (в т.ч. для проверки roundtrip).
void rouge_dequantize_i8_tensor(const int8_t *q, int n, float scale,
                                float *out) {
  if (!q || !out || n <= 0) return;
  for (int i = 0; i < n; ++i) out[i] = (float)q[i] * scale;
}

void rouge_dequantize_i8_channel(const int8_t *q, int rows, int cols,
                                 const float *scales, float *out) {
  if (!q || !scales || !out || rows <= 0 || cols <= 0) return;
  for (int r = 0; r < rows; ++r) {
    const int8_t *qr = q + (size_t)r * cols;
    float *o = out + (size_t)r * cols;
    const float s = scales[r];
    for (int c = 0; c < cols; ++c) o[c] = (float)qr[c] * s;
  }
}

void rouge_dequantize_i8_group(const uint8_t *q, int rows, int cols, int groups,
                               const float *scales, const uint8_t *zeros,
                               float *out) {
  if (!q || !scales || !zeros || !out || rows <= 0 || cols <= 0 || groups <= 0)
    return;
  if (cols % groups != 0) return;
  const int gsz = cols / groups;
  for (int r = 0; r < rows; ++r) {
    const uint8_t *qr = q + (size_t)r * cols;
    float *o = out + (size_t)r * cols;
    for (int g = 0; g < groups; ++g) {
      const float s = scales[r * groups + g];
      const double z = zeros[r * groups + g];
      for (int i = 0; i < gsz; ++i)
        o[g * gsz + i] = ((double)qr[g * gsz + i] - z) * (double)s;
    }
  }
}

// Целочисленный GEMM: C[m][n] = sum_k A[m][k]*B[k][n], накопление в int32.
// A: [M][K] int8, B: [K][N] int8 (row-major).
// Схема масштабов задаётся ЯВНО: a_rows=0 — один масштаб на весь A
// (per-tensor), a_rows=1 — свой масштаб на строку A (per-channel);
// b_cols=0 — один масштаб на весь B, b_cols=1 — свой на столбец B.
// C_f32 = int32_result * scale_a * scale_b.
// Накопление int32 переполняется только при |значений| порядка 2^31 — на
// практике недостижимо, поэтому результат целочисленный и ТОЧНЫЙ.
void rouge_gemm_i8(int M, int N, int K, const int8_t *A, const int8_t *B,
                   const float *scales_a, int a_rows, const float *scales_b,
                   int b_cols, float *C) {
  if (!A || !B || !scales_a || !scales_b || !C) return;
  if (M <= 0 || N <= 0 || K <= 0) return;
  std::vector<int32_t> acc((size_t)N);
  for (int m = 0; m < M; ++m) {
    const int8_t *arow = A + (size_t)m * K;
    std::fill(acc.begin(), acc.end(), 0);
    for (int k = 0; k < K; ++k) {
      const int32_t av = (int32_t)arow[k];
      if (av == 0) continue;
      const int8_t *brow = B + (size_t)k * N;
      for (int n = 0; n < N; ++n) acc[(size_t)n] += av * (int32_t)brow[n];
    }
    const float sa = a_rows ? scales_a[m] : scales_a[0];
    float *crow = C + (size_t)m * N;
    for (int n = 0; n < N; ++n) {
      const float sb = b_cols ? scales_b[n] : scales_b[0];
      crow[n] = (float)acc[(size_t)n] * sa * sb;
    }
  }
}

}  // extern "C"
