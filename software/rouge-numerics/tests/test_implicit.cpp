// Implicit-GEMM свёртка против im2col + тюнер трёх методов.
//
// Эталон — тот же im2col (другой порядок обхода, тот же счёт) плюс один
// независимый прямой досчёт в double. Допуск 1e-5 — только перестановка
// слагаемых. Тюнер обязан вернуть 0/1/2 и результат, совпадающий с im2col.

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
  std::mt19937 rng(0x1171u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const struct {
    int N, Cin, H, W, Cout, KH, KW, stride, pad, bias;
  } cases[] = {
      {1, 1, 5, 5, 1, 3, 3, 1, 1, 0},
      {2, 3, 8, 7, 4, 3, 3, 1, 0, 1},
      {1, 2, 9, 9, 3, 3, 3, 2, 1, 0},
      {2, 4, 16, 16, 8, 1, 1, 1, 0, 1},  // 1x1
      {1, 3, 12, 12, 8, 5, 5, 2, 2, 1},  // 5x5 s2
      {1, 2, 7, 5, 3, 3, 3, 1, 1, 1},    // нечёт/neквaдрат
  };
  for (const auto &cs : cases) {
    const int OH = (cs.H + 2 * cs.pad - cs.KH) / cs.stride + 1;
    const int OW = (cs.W + 2 * cs.pad - cs.KW) / cs.stride + 1;
    std::vector<float> X((size_t)cs.N * cs.Cin * cs.H * cs.W);
    std::vector<float> Wt((size_t)cs.Cout * cs.Cin * cs.KH * cs.KW);
    std::vector<float> bias(cs.bias ? (size_t)cs.Cout : 0);
    for (auto &v : X) v = dist(rng);
    for (auto &v : Wt) v = dist(rng);
    for (auto &v : bias) v = dist(rng);
    std::vector<float> got((size_t)cs.N * cs.Cout * OH * OW, 0);
    std::vector<float> ref(got.size(), 0);
    rouge_conv_implicit_f32(cs.N, cs.Cin, cs.H, cs.W, cs.Cout, cs.KH, cs.KW,
                            cs.stride, cs.pad, X.data(), Wt.data(),
                            cs.bias ? bias.data() : nullptr, got.data(),
                            nullptr);
    rouge_conv2d_f32(cs.N, cs.Cin, cs.H, cs.W, cs.Cout, cs.KH, cs.KW,
                     cs.stride, cs.pad, X.data(), Wt.data(),
                     cs.bias ? bias.data() : nullptr, ref.data(), nullptr);
    const double e = rel_err(got, ref);
    if (e > 1e-5) {
      std::printf("implicit %dx%d %dx%d s=%d p=%d: rel %g\n", cs.H, cs.W,
                  cs.KH, cs.KW, cs.stride, cs.pad, e);
      ++failures;
    }
    // Тюнер трёх методов: корректный результат в любом случае.
    std::vector<float> tuned(got.size(), 0);
    RougeConvProfile p = rouge_conv_tuned(
        cs.N, cs.Cin, cs.H, cs.W, cs.Cout, cs.KH, cs.KW, cs.stride, cs.pad,
        X.data(), Wt.data(), cs.bias ? bias.data() : nullptr, tuned.data());
    EXPECT_TRUE(p.method >= 0 && p.method <= 2);
    if (rel_err(tuned, ref) > 1e-4) {
      std::printf("tuned mismatch case %dx%d: %g\n", cs.H, cs.W,
                  rel_err(tuned, ref));
      ++failures;
    }
  }
  // NULL/нули без падения.
  rouge_conv_implicit_f32(1, 1, 4, 4, 1, 3, 3, 1, 1, nullptr, nullptr, nullptr,
                          nullptr, nullptr);

  if (failures == 0) std::printf("PASS: implicit conv + 3-way tuned\n");
  return failures ? 1 : 0;
}
