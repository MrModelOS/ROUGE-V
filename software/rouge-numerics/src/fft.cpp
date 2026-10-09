/*
 * FFT Wave 2: свой radix-2 Cooley-Tukey, complex interleaved (re,im подряд
 * в RougeComplex). Написан с нуля по общеизвестному алгоритму:
 * бит-реверсная перестановка + итеративные бабочки.
 *
 * Соглашения: forward (inverse=0) — exp(-2πi/N), без нормировки;
 * inverse (!=0) — exp(+2πi/N) с делением на N.
 * n — любой >= 1: степени двойки идут быстрым путём O(N log N), остальные —
 * прямым O(N²) DFT тем же определением (медленно, но корректно; для
 * простых/неудобных размеров честный медленный путь вместо отказа).
 * n <= 0 или NULL -> возврат -1 без записи; успех -> 0.
 */

#include "rouge/numerics.h"

#include <cmath>

namespace {

int is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

// Прямой DFT тем же определением (медленный путь для не-степеней двойки).
void direct_dft(int n, const RougeComplex *in, RougeComplex *out, int inverse) {
  const double sgn = inverse ? 1.0 : -1.0;
  const double norm = inverse ? 1.0 / n : 1.0;
  for (int k = 0; k < n; ++k) {
    double re = 0.0, im = 0.0;
    for (int j = 0; j < n; ++j) {
      const double a = sgn * 2.0 * 3.14159265358979323846 * k * j / n;
      const double c = std::cos(a), s = std::sin(a);
      re += in[j].re * c - in[j].im * s;
      im += in[j].re * s + in[j].im * c;
    }
    out[k].re = (float)(re * norm);
    out[k].im = (float)(im * norm);
  }
}

}  // namespace

extern "C" {

int rouge_fft_f32(int n, const RougeComplex *in, RougeComplex *out,
                  int inverse) {
  if (!in || !out || n <= 0) return -1;
  if (!is_pow2(n)) {
    direct_dft(n, in, out, inverse);
    return 0;
  }
  for (int i = 0; i < n; ++i) out[i] = in[i];
  // Бит-реверс.
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      RougeComplex t = out[i];
      out[i] = out[j];
      out[j] = t;
    }
  }
  const float sign = inverse ? 1.0f : -1.0f;
  for (int len = 2; len <= n; len <<= 1) {
    const float ang = sign * 2.0f * 3.14159265358979323846f / (float)len;
    const float wlen_re = std::cos(ang), wlen_im = std::sin(ang);
    for (int i = 0; i < n; i += len) {
      float w_re = 1.0f, w_im = 0.0f;
      for (int j = 0; j < len / 2; ++j) {
        const float u_re = out[i + j].re, u_im = out[i + j].im;
        const float v_re =
            out[i + j + len / 2].re * w_re - out[i + j + len / 2].im * w_im;
        const float v_im =
            out[i + j + len / 2].re * w_im + out[i + j + len / 2].im * w_re;
        out[i + j].re = u_re + v_re;
        out[i + j].im = u_im + v_im;
        out[i + j + len / 2].re = u_re - v_re;
        out[i + j + len / 2].im = u_im - v_im;
        const float nw_re = w_re * wlen_re - w_im * wlen_im;
        const float nw_im = w_re * wlen_im + w_im * wlen_re;
        w_re = nw_re;
        w_im = nw_im;
      }
    }
  }
  if (inverse) {
    const float inv = 1.0f / (float)n;
    for (int i = 0; i < n; ++i) {
      out[i].re *= inv;
      out[i].im *= inv;
    }
  }
  return 0;
}

}  // extern "C"
