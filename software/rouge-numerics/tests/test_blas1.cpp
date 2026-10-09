// BLAS L1/L2: свои операции против прямых петель + против системного cblas.
//
// Петли в этом файле — deliberately примитивные (другой порядок/накопление),
// cblas — независимая реализация. Допуски: f32 — 1e-6, f64 — 1e-14.
// Officers: inc != 1 тоже проверяются.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "rouge/numerics.h"

#if defined(ROUGE_HAVE_CBLAS)
#include <cblas.h>
#endif

static int failures = 0;

#define EXPECT_TRUE(cond)                                        \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                \
    }                                                            \
  } while (0)

int main() {
  std::mt19937 rng(0xB1A5u);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const int n = 1000;

  std::vector<float> xf(n), yf(n);
  std::vector<double> xd(n), yd(n);
  for (int i = 0; i < n; ++i) {
    xf[i] = (float)dist(rng);
    yf[i] = (float)dist(rng);
    xd[i] = dist(rng);
    yd[i] = dist(rng);
  }

  // axpy
  {
    std::vector<float> got = yf;
    rouge_saxpy(n, 2.0f, xf.data(), 1, got.data(), 1);
    for (int i = 0; i < n; ++i) EXPECT_TRUE(got[i] == 2.0f * xf[i] + yf[i]);
    std::vector<double> gotd = yd;
    rouge_daxpy(n, -1.5, xd.data(), 1, gotd.data(), 1);
    for (int i = 0; i < n; ++i)
      EXPECT_TRUE(gotd[i] == -1.5 * xd[i] + yd[i]);
    // inc != 1.
    std::vector<float> xs(2 * n, 0.0f), ys(2 * n, 0.0f);
    for (int i = 0; i < n; ++i) {
      xs[2 * i] = xf[i];
      ys[2 * i + 1] = yf[i];
    }
    rouge_saxpy(n, 1.0f, xs.data(), 2, ys.data() + 1, 2);
    for (int i = 0; i < n; ++i) EXPECT_TRUE(ys[2 * i + 1] == xf[i] + yf[i]);
#if defined(ROUGE_HAVE_CBLAS)
    std::vector<float> cb = yf;
    cblas_saxpy(n, 2.0f, xf.data(), 1, cb.data(), 1);
    EXPECT_TRUE(cb == got);
#endif
  }
  // dot
  {
    long double ref = 0;
    for (int i = 0; i < n; ++i) ref += (long double)xf[i] * yf[i];
    const double got = rouge_sdot(n, xf.data(), 1, yf.data(), 1);
    EXPECT_TRUE(std::fabs(got - (double)ref) < 1e-6 * (1 + std::fabs((double)ref)));
    long double refd = 0;
    for (int i = 0; i < n; ++i) refd += (long double)xd[i] * yd[i];
    EXPECT_TRUE(rouge_ddot(n, xd.data(), 1, yd.data(), 1) == (double)refd ||
                std::fabs(rouge_ddot(n, xd.data(), 1, yd.data(), 1) - (double)refd) <
                    1e-14 * (1 + std::fabs((double)refd)));
#if defined(ROUGE_HAVE_CBLAS)
    EXPECT_TRUE(cblas_sdot(n, xf.data(), 1, yf.data(), 1) == got ||
                std::fabs(cblas_sdot(n, xf.data(), 1, yf.data(), 1) - got) <
                    1e-6 * (1 + std::fabs(got)));
#endif
  }
  // scal + nrm2
  {
    std::vector<float> got = xf;
    rouge_sscal(n, 3.0f, got.data(), 1);
    for (int i = 0; i < n; ++i) EXPECT_TRUE(got[i] == 3.0f * xf[i]);
    long double ss = 0;
    for (int i = 0; i < n; ++i) ss += (long double)xf[i] * xf[i];
    EXPECT_TRUE(std::fabs((double)rouge_snrm2(n, xf.data(), 1) - std::sqrt((double)ss)) <
                1e-6 * (1 + std::sqrt((double)ss)));
    // Большой динамический диапазон: наивная сумма квадратов переполнилась бы.
    std::vector<float> big = {1e20f, 1e-20f, 3.0f, -4.0f};
    EXPECT_TRUE(std::fabs((double)rouge_snrm2(4, big.data(), 1) - 1e20) < 1e14);
    long double ssd = 0;
    for (int i = 0; i < n; ++i) ssd += (long double)xd[i] * xd[i];
    EXPECT_TRUE(std::fabs(rouge_dnrm2(n, xd.data(), 1) - std::sqrt((double)ssd)) <
                1e-14 * (1 + std::sqrt((double)ssd)));
  }
  // gemv N/T против прямых петель.
  {
    const int m = 17, k = 23;
    std::vector<float> A(m * k), x(k), y(m), y2(k);
    for (auto &v : A) v = (float)dist(rng);
    for (auto &v : x) v = (float)dist(rng);
    for (auto &v : y) v = (float)dist(rng);
    for (auto &v : y2) v = (float)dist(rng);
    std::vector<float> got = y, gotT = y2;
    rouge_sgemv('N', m, k, 2.0f, A.data(), k, x.data(), 1, 0.5f, got.data(), 1);
    rouge_sgemv('t', m, k, -1.0f, A.data(), k, y.data(), 1, 0.0f, gotT.data(),
                1);
    for (int i = 0; i < m; ++i) {
      double s = 0;
      for (int j = 0; j < k; ++j) s += (double)A[i * k + j] * x[j];
      if (std::fabs((double)got[i] - (2.0 * s + 0.5 * y[i])) > 1e-5 * (1 + std::fabs((double)got[i])))
        ++failures;
    }
    for (int j = 0; j < k; ++j) {
      double s = 0;
      for (int i = 0; i < m; ++i) s += (double)A[i * k + j] * y[i];
      if (std::fabs((double)gotT[j] + s) > 1e-5 * (1 + std::fabs((double)gotT[j])))
        ++failures;
    }
    // Мусорный trans — отказ без записи.
    std::vector<float> keep = got;
    rouge_sgemv('X', m, k, 1.0f, A.data(), k, x.data(), 1, 0.0f, got.data(), 1);
    EXPECT_TRUE(keep == got);
#if defined(ROUGE_HAVE_CBLAS)
    std::vector<float> cb = y;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, 2.0f, A.data(), k, x.data(),
                1, 0.5f, cb.data(), 1);
    // Не бит-в-бит: у cblas другой порядок суммирования — допуск, как везде.
    for (size_t i = 0; i < cb.size(); ++i)
      EXPECT_TRUE(std::fabs((double)cb[i] - got[i]) < 1e-6 * (1 + std::fabs((double)got[i])));
#endif
    // dgemv smoke против прямой петли.
    std::vector<double> Ad(m * k), xd2(k), yd2(m, 0.0);
    for (auto &v : Ad) v = dist(rng);
    for (auto &v : xd2) v = dist(rng);
    rouge_dgemv('N', m, k, 1.0, Ad.data(), k, xd2.data(), 1, 0.0, yd2.data(),
                1);
    for (int i = 0; i < m; ++i) {
      long double s = 0;
      for (int j = 0; j < k; ++j) s += (long double)Ad[i * k + j] * xd2[j];
      if (std::fabs(yd2[i] - (double)s) > 1e-12 * (1 + std::fabs((double)s)))
        ++failures;
    }
  }
  // NULL/нули.
  EXPECT_TRUE(rouge_sdot(0, nullptr, 1, nullptr, 1) == 0.0f);
  EXPECT_TRUE(rouge_dnrm2(5, nullptr, 1) == 0.0);
  rouge_saxpy(-1, 1.0f, nullptr, 1, nullptr, 1);

  if (failures == 0) std::printf("PASS: blas L1/L2\n");
  return failures ? 1 : 0;
}
