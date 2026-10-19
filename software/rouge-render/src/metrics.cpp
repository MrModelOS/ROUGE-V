/*
 * Метрики качества: PSNR и SSIM.
 *
 * Считаются по «печатному» изображению (после тонового отображения) — так
 * же, как это делают в работах по реконструкции: сравнивать линейный HDR
 * бессмысленно, потому что глаз видит тонированную картинку.
 *
 * SSIM — стандартная формула (Wang et al. 2004) на яркости, окно 11x11
 * гаусс sigma=1.5, C1=(0.01)^2, C2=(0.03)^2 (для диапазона [0,1]).
 */

#include "rouge/render.h"

#include <cmath>
#include <vector>

namespace {

inline float luma(const float *p) {
  return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
}

}  // namespace

extern "C" {

double rouge_psnr(const RougeImage *a, const RougeImage *b) {
  if (!a || !b) return 0.0;
  if (a->w != b->w || a->h != b->h) return 0.0;
  const size_t n = (size_t)a->w * a->h;
  if (n == 0) return 0.0;
  double se = 0.0;
  for (size_t i = 0; i < n; ++i) {
    for (int c = 0; c < 3; ++c) {
      const double d = (double)a->px[i * 3 + c] - (double)b->px[i * 3 + c];
      se += d * d;
    }
  }
  const double mse = se / (double)(n * 3);
  if (mse <= 0.0) return 99.0;
  return 10.0 * std::log10(1.0 / mse);
}

double rouge_ssim(const RougeImage *a, const RougeImage *b) {
  if (!a || !b) return 0.0;
  if (a->w != b->w || a->h != b->h) return 0.0;
  const int w = a->w, h = a->h;
  if (w < 11 || h < 11) return 0.0;

  // Гауссово окно 11x11, sigma=1.5, нормированное.
  const int R = 5;
  double k[11];
  double ksum = 0.0;
  for (int i = -R; i <= R; ++i) {
    const double v = std::exp(-(double)(i * i) / (2.0 * 1.5 * 1.5));
    k[i + R] = v;
    ksum += v;
  }
  for (int i = 0; i < 11; ++i) k[i] /= ksum;

  std::vector<float> ya((size_t)w * h), yb((size_t)w * h);
  for (size_t i = 0; i < (size_t)w * h; ++i) {
    ya[i] = luma(&a->px[i * 3]);
    yb[i] = luma(&b->px[i * 3]);
  }

  const double C1 = 0.01 * 0.01;
  const double C2 = 0.03 * 0.03;
  double total = 0.0;
  int count = 0;
  for (int y = 0; y + 2 * R < h; ++y) {
    for (int x = 0; x + 2 * R < w; ++x) {
      double mu_a = 0.0, mu_b = 0.0;
      for (int j = -R; j <= R; ++j)
        for (int i = -R; i <= R; ++i) {
          const double wgt = k[i + R] * k[j + R];
          mu_a += wgt * ya[(size_t)(y + j) * w + (x + i)];
          mu_b += wgt * yb[(size_t)(y + j) * w + (x + i)];
        }
      double saa = 0.0, sbb = 0.0, sab = 0.0;
      for (int j = -R; j <= R; ++j)
        for (int i = -R; i <= R; ++i) {
          const double wgt = k[i + R] * k[j + R];
          const double da = ya[(size_t)(y + j) * w + (x + i)] - mu_a;
          const double db = yb[(size_t)(y + j) * w + (x + i)] - mu_b;
          saa += wgt * da * da;
          sbb += wgt * db * db;
          sab += wgt * da * db;
        }
      const double s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) /
                       ((mu_a * mu_a + mu_b * mu_b + C1) *
                        (saa + sbb + C2));
      total += s;
      ++count;
    }
  }
  if (count == 0) return 0.0;
  return total / (double)count;
}

}  // extern "C"
