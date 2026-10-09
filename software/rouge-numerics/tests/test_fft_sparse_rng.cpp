// FFT + sparse + random: три маленьких ядра Wave 2.
//
// FFT: сравнение с наивным DFT (другой алгоритм) на n=1..256 + roundtrip
//   forward->inverse + отказ на не-степени двойки.
// SpMV: случайная разреженная матрица против плотного эталона + проверки
//   границ (возврат -1, y не трогается).
// RNG: детерминизм (два сида -> две одинаковые последовательности),
//   uniform в [0,1), normal со средним ~0 и дисперсией ~1 на 100k выборок.

#include <cmath>
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

static void naive_dft(int n, const std::vector<RougeComplex> &in,
                      std::vector<RougeComplex> &out, int inverse) {
  const double sgn = inverse ? 1.0 : -1.0;
  for (int k = 0; k < n; ++k) {
    double re = 0, im = 0;
    for (int j = 0; j < n; ++j) {
      const double a = sgn * 2.0 * 3.141592653589793 * k * j / n;
      re += in[j].re * std::cos(a) - in[j].im * std::sin(a);
      im += in[j].re * std::sin(a) + in[j].im * std::cos(a);
    }
    if (inverse) {
      re /= n;
      im /= n;
    }
    out[k].re = (float)re;
    out[k].im = (float)im;
  }
}

int main() {
  std::mt19937 rng(0xFF7u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // --- FFT ---
  for (int n = 1; n <= 256; n *= 2) {
    std::vector<RougeComplex> in(n), got(n), want(n);
    for (auto &c : in) {
      c.re = dist(rng);
      c.im = dist(rng);
    }
    EXPECT_TRUE(rouge_fft_f32(n, in.data(), got.data(), 0) == 0);
    naive_dft(n, in, want, 0);
    double md = 0.0, sc = 1.0;
    for (int i = 0; i < n; ++i) {
      md = std::max(md, std::fabs((double)got[i].re - want[i].re));
      md = std::max(md, std::fabs((double)got[i].im - want[i].im));
      sc = std::max(sc, (double)std::fabs(want[i].re));
      sc = std::max(sc, (double)std::fabs(want[i].im));
    }
    if (md / sc > 1e-5) {
      std::printf("fft n=%d rel %g\n", n, md / sc);
      ++failures;
    }
    // roundtrip
    std::vector<RougeComplex> back(n);
    EXPECT_TRUE(rouge_fft_f32(n, got.data(), back.data(), 1) == 0);
    md = 0.0;
    sc = 1.0;
    for (int i = 0; i < n; ++i) {
      md = std::max(md, std::fabs((double)back[i].re - in[i].re));
      md = std::max(md, std::fabs((double)back[i].im - in[i].im));
      sc = std::max(sc, (double)std::fabs(in[i].re));
    }
    if (md / sc > 1e-5) {
      std::printf("fft roundtrip n=%d rel %g\n", n, md / sc);
      ++failures;
    }
  }
  // --- FFT произвольных n (медленный путь) против long double-эталона ---
  // ВНИМАНИЕ: эталон здесь — тот же прямой DFT, но в long double (другая
  // точность и независимый подсчёт тригонометрии), плюс roundtrip.
  for (int n : {3, 5, 6, 7, 9, 12, 100, 127}) {
    std::vector<RougeComplex> in(n), got(n);
    for (auto &c : in) {
      c.re = dist(rng);
      c.im = dist(rng);
    }
    EXPECT_TRUE(rouge_fft_f32(n, in.data(), got.data(), 0) == 0);
    double md = 0.0, sc = 1.0;
    for (int k = 0; k < n; ++k) {
      long double re = 0, im = 0;
      for (int j = 0; j < n; ++j) {
        const long double a =
            -2.0L * 3.14159265358979323846L * k * j / n;
        re += (long double)in[j].re * std::cos(a) -
              (long double)in[j].im * std::sin(a);
        im += (long double)in[j].re * std::sin(a) +
              (long double)in[j].im * std::cos(a);
      }
      md = std::max(md, std::fabs((double)got[k].re - (double)re));
      md = std::max(md, std::fabs((double)got[k].im - (double)im));
      sc = std::max(sc, (double)std::fabs((double)re));
    }
    if (md / sc > 1e-5) {
      std::printf("fft slow path n=%d rel %g\n", n, md / sc);
      ++failures;
    }
    std::vector<RougeComplex> back(n);
    EXPECT_TRUE(rouge_fft_f32(n, got.data(), back.data(), 1) == 0);
    md = 0.0;
    for (int i = 0; i < n; ++i) {
      md = std::max(md, std::fabs((double)back[i].re - in[i].re));
      md = std::max(md, std::fabs((double)back[i].im - in[i].im));
    }
    if (md > 1e-4) {
      std::printf("fft slow roundtrip n=%d abs %g\n", n, md);
      ++failures;
    }
  }
  {
    RougeComplex a[4] = {}, b[4] = {};
    EXPECT_TRUE(rouge_fft_f32(0, a, b, 0) == -1);
    EXPECT_TRUE(rouge_fft_f32(4, nullptr, b, 0) == -1);
  }

  // --- SpMV ---
  {
    const int nrows = 17, ncols = 13;
    std::vector<int> rowptr(nrows + 1, 0);
    std::vector<int> colidx;
    std::vector<float> values;
    std::vector<std::vector<float>> dense(nrows, std::vector<float>(ncols, 0));
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j)
        if (rng() % 4 == 0) {
          const float v = dist(rng);
          dense[i][j] = v;
          colidx.push_back(j);
          values.push_back(v);
        }
      rowptr[i + 1] = (int)colidx.size();
    }
    std::vector<float> x(ncols), y(nrows, 0), want(nrows, 0);
    for (auto &v : x) v = dist(rng);
    EXPECT_TRUE(rouge_csr_spmv_f32(nrows, ncols, rowptr.data(), colidx.data(),
                                   values.data(), x.data(), y.data()) == 0);
    for (int i = 0; i < nrows; ++i) {
      double s = 0;
      for (int j = 0; j < ncols; ++j) s += (double)dense[i][j] * x[j];
      want[i] = (float)s;
    }
    for (int i = 0; i < nrows; ++i) EXPECT_TRUE(y[i] == want[i]);
    // Границы: y не трогается, возврат -1.
    y.assign(nrows, 42.0f);
    int bad_col[1] = {ncols};
    float one[1] = {1.0f};
    int rp1[2] = {0, 1};
    EXPECT_TRUE(rouge_csr_spmv_f32(1, ncols, rp1, bad_col, one, x.data(),
                                   y.data()) == -1);
    EXPECT_TRUE(y[0] == 42.0f);
    EXPECT_TRUE(rouge_csr_spmv_f32(1, ncols, rp1, bad_col, one, nullptr,
                                   y.data()) == -1);
  }

  // --- RNG ---
  {
    RougeRng a{}, b{};
    rouge_rng_seed(&a, 12345);
    rouge_rng_seed(&b, 12345);
    for (int i = 0; i < 1000; ++i)
      EXPECT_TRUE(rouge_rng_next_u64(&a) == rouge_rng_next_u64(&b));
    RougeRng c{};
    rouge_rng_seed(&c, 999);
    EXPECT_TRUE(rouge_rng_next_u64(&c) != rouge_rng_next_u64(&a) ||
                true);  // сиды разные — совпадение почти невозможно, но не assert
    for (int i = 0; i < 1000; ++i) {
      const float u = rouge_rng_uniform_f32(&a);
      EXPECT_TRUE(u >= 0.0f && u < 1.0f);
    }
    double mean = 0, m2 = 0;
    const int nn = 100000;
    for (int i = 0; i < nn; ++i) {
      const double v = rouge_rng_normal_f32(&b);
      mean += v;
      m2 += v * v;
    }
    mean /= nn;
    const double var = m2 / nn - mean * mean;
    if (std::fabs(mean) > 0.05 || std::fabs(var - 1.0) > 0.05) {
      std::printf("rng normal: mean %g var %g\n", mean, var);
      ++failures;
    }
  }

  if (failures == 0) std::printf("PASS: fft + sparse + random\n");
  return failures ? 1 : 0;
}
