// Winograd F(2x2,3x3) против прямого эталона + автотюнер свёртки.
//
// Эталон — прямые петли (другой алгоритм). Случайные конфигурации: чётные и
// нечётные H/W (хвосты), bias on/off. Допуск 1e-4 — арифметика Winograd
// переставляет слагаемые, строгого бит-совпадения нет и не обещается.

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

static double rel_err(const std::vector<float> &got,
                      const std::vector<float> &want) {
  double md = 0.0, sc = 1.0;
  for (size_t i = 0; i < want.size(); ++i) {
    md = std::max(md, std::fabs((double)got[i] - (double)want[i]));
    sc = std::max(sc, std::fabs((double)want[i]));
  }
  return md / sc;
}

int main() {
  std::mt19937 rng(0x90170u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const struct {
    int N, Cin, H, W, Cout, bias;
  } cases[] = {
      {1, 1, 4, 4, 1, 0}, {1, 2, 6, 6, 2, 1}, {2, 3, 8, 8, 4, 1},
      {1, 1, 5, 7, 1, 0},  // нечётные — хвосты прямым досчётом
      {1, 4, 16, 16, 8, 1},
  };
  // Фиксированный сид без знаковых трюков.
  rng.seed(0x90170u);
  for (const auto &cs : cases) {
    std::vector<float> X((size_t)cs.N * cs.Cin * cs.H * cs.W);
    std::vector<float> Wt((size_t)cs.Cout * cs.Cin * 9);
    std::vector<float> bias(cs.bias ? (size_t)cs.Cout : 0);
    for (auto &v : X) v = dist(rng);
    for (auto &v : Wt) v = dist(rng);
    for (auto &v : bias) v = dist(rng);
    std::vector<float> wino((size_t)cs.N * cs.Cout * cs.H * cs.W, 0);
    std::vector<float> ref(wino.size(), 0);
    int rc = rouge_conv_winograd3x3_f32(cs.N, cs.Cin, cs.H, cs.W, cs.Cout,
                                        X.data(), Wt.data(),
                                        cs.bias ? bias.data() : nullptr,
                                        wino.data());
    EXPECT_TRUE(rc == 0);
    rouge_conv2d_f32(cs.N, cs.Cin, cs.H, cs.W, cs.Cout, 3, 3, 1, 1, X.data(),
                     Wt.data(), cs.bias ? bias.data() : nullptr, ref.data(),
                     nullptr);
    const double e = rel_err(wino, ref);
    if (e > 1e-4) {
      std::printf("winograd case N=%d Cin=%d %dx%d Cout=%d bias=%d: rel %g\n",
                  cs.N, cs.Cin, cs.H, cs.W, cs.Cout, cs.bias, e);
      ++failures;
    }
  }

  // Ручной случай: единичный вход 4x4, ядро с единицей в центре -> выход == вход.
  {
    std::vector<float> X(16);
    for (int i = 0; i < 16; ++i) X[i] = (float)(i + 1);
    std::vector<float> Wt(9, 0.0f);
    Wt[4] = 1.0f;
    std::vector<float> got(16, 0), ref(16, 0);
    EXPECT_TRUE(rouge_conv_winograd3x3_f32(1, 1, 4, 4, 1, X.data(), Wt.data(),
                                           nullptr, got.data()) == 0);
    rouge_conv2d_f32(1, 1, 4, 4, 1, 3, 3, 1, 1, X.data(), Wt.data(), nullptr,
                     ref.data(), nullptr);
    EXPECT_TRUE(rel_err(got, ref) < 1e-5);
    EXPECT_TRUE(rel_err(got, X) < 1e-5);
  }

  // Автотюнер: корректный результат + замеры обоих методов без падения.
  {
    const int N = 1, Cin = 2, H = 8, W = 8, Cout = 2;
    std::vector<float> X((size_t)N * Cin * H * W), Wt((size_t)Cout * Cin * 9),
        bias(Cout, 0.5f);
    for (auto &v : X) v = dist(rng);
    for (auto &v : Wt) v = dist(rng);
    std::vector<float> got((size_t)N * Cout * H * W, 0), ref(got.size(), 0);
    RougeConvProfile p = rouge_conv_tuned(N, Cin, H, W, Cout, 3, 3, 1, 1,
                                          X.data(), Wt.data(), bias.data(),
                                          got.data());
    EXPECT_TRUE(p.method == 0 || p.method == 1);
    rouge_conv2d_f32(N, Cin, H, W, Cout, 3, 3, 1, 1, X.data(), Wt.data(),
                     bias.data(), ref.data(), nullptr);
    if (rel_err(got, ref) > 1e-4) {
      std::printf("conv_tuned mismatch: %g\n", rel_err(got, ref));
      ++failures;
    }
    std::printf("conv_tuned: method=%d (0=im2col,1=winograd) %.3f ms\n",
                p.method, p.millis);
    // Неприменимый случай (5x5) — только im2col.
    std::vector<float> W5((size_t)Cout * Cin * 25, 0.25f);
    RougeConvProfile p2 = rouge_conv_tuned(N, Cin, H, W, Cout, 5, 5, 1, 2,
                                           X.data(), W5.data(), bias.data(),
                                           got.data());
    EXPECT_TRUE(p2.method == 0);
  }

  if (failures == 0) std::printf("PASS: winograd + conv tuned\n");
  return failures ? 1 : 0;
}
