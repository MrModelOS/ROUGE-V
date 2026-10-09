// Свёртка Wave 2: im2col+GEMM против независимого прямого эталона.
//
// Эталон в этом файле — прямые вложенные петли (другой алгоритм, чем
// тестируемый im2col+GEMM), плюс один ручной случай 1x1x3x3 / 1x3x3,
// посчитанный на бумаге. Случайные размеры покрывают stride 1-2, pad 0-1,
// некратные плитки и bias on/off.

#include <cmath>
#include <cstdint>
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

// Прямой эталон: сумма по Cin*KH*KW с нулевым паддингом за краями.
static void ref_conv(int N, int Cin, int H, int W, int Cout, int KH, int KW,
                     int stride, int pad, const std::vector<float> &X,
                     const std::vector<float> &Wt,
                     const std::vector<float> &bias, std::vector<float> &Y) {
  const int OH = (H + 2 * pad - KH) / stride + 1;
  const int OW = (W + 2 * pad - KW) / stride + 1;
  for (int n = 0; n < N; ++n)
    for (int co = 0; co < Cout; ++co)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
          double s = bias.empty() ? 0.0 : bias[co];
          for (int ci = 0; ci < Cin; ++ci)
            for (int kr = 0; kr < KH; ++kr)
              for (int kc = 0; kc < KW; ++kc) {
                const int ih = oh * stride - pad + kr;
                const int iw = ow * stride - pad + kc;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                s += (double)X[((size_t)n * Cin + ci) * H * W + ih * W + iw] *
                     (double)Wt[((size_t)co * Cin + ci) * KH * KW + kr * KW +
                                kc];
              }
          Y[((size_t)n * Cout + co) * OH * OW + oh * OW + ow] = (float)s;
        }
}

static double rel_error(const std::vector<float> &got,
                        const std::vector<float> &want) {
  double maxdiff = 0.0, scale = 1.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = std::fabs((double)got[i] - (double)want[i]);
    if (d > maxdiff) maxdiff = d;
    if (std::fabs((double)want[i]) > scale) scale = std::fabs((double)want[i]);
  }
  return maxdiff / scale;
}

int main() {
  // --- 1. ручной случай: X=1..9 (3x3), ядро [[1,0,-1],[1,0,-1],[1,0,-1]],
  // stride 1, pad 0 -> один выход: (1+4+7) - (3+6+9) = -6. ---
  {
    const std::vector<float> X = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<float> Wt = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    const std::vector<float> bias;
    std::vector<float> Y(1, 0), ref(1, 0);
    rouge_conv2d_f32(1, 1, 3, 3, 1, 3, 3, 1, 0, X.data(), Wt.data(), nullptr,
                     Y.data(), nullptr);
    ref_conv(1, 1, 3, 3, 1, 3, 3, 1, 0, X, Wt, bias, ref);
    EXPECT_TRUE(Y[0] == -6.0f && ref[0] == -6.0f);
  }

  // --- 2. случайные конфигурации против прямого эталона ---
  std::mt19937 rng(0xC0FFEEu);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const struct {
    int N, Cin, H, W, Cout, KH, KW, stride, pad, useBias;
  } cases[] = {
      {1, 1, 5, 5, 1, 3, 3, 1, 1, 1},
      {2, 3, 8, 7, 4, 3, 3, 1, 0, 1},
      {1, 2, 9, 9, 3, 3, 3, 2, 1, 0},
      {2, 4, 16, 16, 8, 1, 1, 1, 0, 1},   // 1x1
      {1, 3, 32, 32, 8, 5, 5, 2, 2, 1},   // 5x5, stride 2
      {1, 1, 7, 11, 2, 3, 5, 1, 0, 0},    // неквадратное ядро
  };
  for (const auto &cs : cases) {
    const int OH = (cs.H + 2 * cs.pad - cs.KH) / cs.stride + 1;
    const int OW = (cs.W + 2 * cs.pad - cs.KW) / cs.stride + 1;
    std::vector<float> X((size_t)cs.N * cs.Cin * cs.H * cs.W);
    std::vector<float> Wt((size_t)cs.Cout * cs.Cin * cs.KH * cs.KW);
    std::vector<float> bias(cs.useBias ? (size_t)cs.Cout : 0);
    for (auto &v : X) v = dist(rng);
    for (auto &v : Wt) v = dist(rng);
    for (auto &v : bias) v = dist(rng);
    std::vector<float> got((size_t)cs.N * cs.Cout * OH * OW, 0);
    std::vector<float> want(got.size(), 0);
    rouge_conv2d_f32(cs.N, cs.Cin, cs.H, cs.W, cs.Cout, cs.KH, cs.KW,
                     cs.stride, cs.pad, X.data(), Wt.data(),
                     cs.useBias ? bias.data() : nullptr, got.data(), nullptr);
    ref_conv(cs.N, cs.Cin, cs.H, cs.W, cs.Cout, cs.KH, cs.KW, cs.stride,
             cs.pad, X, Wt, bias, want);
    const double err = rel_error(got, want);
    if (err > 1e-5) {
      std::printf("case N=%d Cin=%d %dx%d Cout=%d %dx%d s=%d p=%d bias=%d: "
                  "rel err %g\n",
                  cs.N, cs.Cin, cs.H, cs.W, cs.Cout, cs.KH, cs.KW, cs.stride,
                  cs.pad, cs.useBias, err);
      ++failures;
    }
  }

  if (failures == 0) std::printf("PASS: conv2d (hand case + 6 random)\n");
  return failures ? 1 : 0;
}
