// Backward свёртки против численного градиента (центральные разности).
//
// Численный градиент — независимое определение производной: никакого цепного
// правила, только forward в двух близких точках. h=1e-3, допуск 1e-3.
// Размеры крошечные (1x1x4x4, ядро 3x3), иначе центральные разности по каждому
// весу/входу считали бы вечность.

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

// Скалярная потеря: сумма выходов forward (bias 0).
static double loss(int N, int Cin, int H, int Wd, int Cout, const float *X,
                   const float *Wt, int KH, int KW) {
  const int OH = H - KH + 1, OW = Wd - KW + 1;
  std::vector<float> Y((size_t)N * Cout * OH * OW, 0);
  rouge_conv2d_f32(N, Cin, H, Wd, Cout, KH, KW, 1, 0, X, Wt, nullptr, Y.data(),
                   nullptr);
  double s = 0;
  for (float v : Y) s += v;
  return s;
}

int main() {
  std::mt19937 rng(0x8D00u);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  const int N = 1, Cin = 1, H = 4, W = 4, Cout = 1, KH = 3, KW = 3;
  const int OH = 2, OW = 2;
  std::vector<float> X(N * Cin * H * W), Wt(Cout * Cin * 9);
  for (auto &v : X) v = dist(rng);
  for (auto &v : Wt) v = dist(rng);
  // dy = dL/dY для L = sum(Y) — все единицы.
  std::vector<float> dy(N * Cout * OH * OW, 1.0f);
  std::vector<float> dx(X.size(), 0), dw(Wt.size(), 0);
  rouge_conv_bwd_data(N, Cin, H, W, Cout, KH, KW, 1, 0, dy.data(), Wt.data(),
                      dx.data());
  rouge_conv_bwd_filter(N, Cin, H, W, Cout, KH, KW, 1, 0, X.data(), dy.data(),
                        dw.data());

  const float h = 1e-3f;
  // dL/dX численно.
  for (size_t i = 0; i < X.size(); ++i) {
    const float old = X[i];
    X[i] = old + h;
    const double lp = loss(N, Cin, H, W, Cout, X.data(), Wt.data(), KH, KW);
    X[i] = old - h;
    const double lm = loss(N, Cin, H, W, Cout, X.data(), Wt.data(), KH, KW);
    X[i] = old;
    const double num = (lp - lm) / (2 * h);
    if (std::fabs((double)dx[i] - num) > 1e-3 * (1 + std::fabs(num))) {
      std::printf("bwd_data @%zu: got %g num %g\n", i, (double)dx[i], num);
      ++failures;
    }
  }
  // dL/dW численно.
  for (size_t i = 0; i < Wt.size(); ++i) {
    const float old = Wt[i];
    Wt[i] = old + h;
    const double lp = loss(N, Cin, H, W, Cout, X.data(), Wt.data(), KH, KW);
    Wt[i] = old - h;
    const double lm = loss(N, Cin, H, W, Cout, X.data(), Wt.data(), KH, KW);
    Wt[i] = old;
    const double num = (lp - lm) / (2 * h);
    if (std::fabs((double)dw[i] - num) > 1e-3 * (1 + std::fabs(num))) {
      std::printf("bwd_filter @%zu: got %g num %g\n", i, (double)dw[i], num);
      ++failures;
    }
  }
  // Stride 2 + pad 1 дымом против численного (меньше точек).
  {
    const int H2 = 5, W2 = 5, OH2 = (H2 + 2 - 3) / 2 + 1, OW2 = (W2 + 2 - 3) / 2 + 1;
    std::vector<float> X2(H2 * W2), W2v(9), dy2(OH2 * OW2, 1.0f), dx2(H2 * W2, 0);
    for (auto &v : X2) v = dist(rng);
    for (auto &v : W2v) v = dist(rng);
    rouge_conv_bwd_data(1, 1, H2, W2, 1, 3, 3, 2, 1, dy2.data(), W2v.data(),
                        dx2.data());
    auto loss2 = [&](const float *Xx) {
      std::vector<float> Y((size_t)OH2 * OW2, 0);
      rouge_conv2d_f32(1, 1, H2, W2, 1, 3, 3, 2, 1, Xx, W2v.data(), nullptr,
                       Y.data(), nullptr);
      double s = 0;
      for (float v : Y) s += v;
      return s;
    };
    for (size_t i = 0; i < X2.size(); ++i) {
      const float old = X2[i];
      X2[i] = old + h;
      const double lp = loss2(X2.data());
      X2[i] = old - h;
      const double lm = loss2(X2.data());
      X2[i] = old;
      if (std::fabs((double)dx2[i] - (lp - lm) / (2 * h)) >
          1e-3 * (1 + std::fabs((lp - lm) / (2 * h))))
        ++failures;
    }
  }
  // NULL без падения.
  rouge_conv_bwd_data(1, 1, 4, 4, 1, 3, 3, 1, 0, nullptr, Wt.data(), dx.data());
  rouge_conv_bwd_filter(1, 1, 4, 4, 1, 3, 3, 1, 0, X.data(), nullptr, dw.data());

  if (failures == 0) std::printf("PASS: conv backward vs numeric gradient\n");
  return failures ? 1 : 0;
}
