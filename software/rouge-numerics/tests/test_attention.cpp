// Attention против наивного двухпроходного эталона (другой алгоритм).
//
// Эталон считает полный softmax в double: сначала max, потом sumexp, потом
// взвешенную сумму. Проверяются: обычный, causal, kv_off, scale по умолчанию,
// большая Skv (несколько блоков Bt=64, пересчёт m/l).

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

static void ref_attn(int S, int Skv, int D, const std::vector<float> &Q,
                     const std::vector<float> &K, const std::vector<float> &V,
                     std::vector<float> &O, float scale, int causal,
                     int kv_off) {
  if (scale <= 0) scale = 1.0f / std::sqrt((float)D);
  for (int s = 0; s < S; ++s) {
    double mx = -1e300;
    for (int t = 0; t < Skv; ++t) {
      if (causal && t > s + kv_off) continue;
      double d = 0;
      for (int k = 0; k < D; ++k)
        d += (double)Q[(size_t)s * D + k] * (double)K[(size_t)t * D + k];
      mx = std::max(mx, d * scale);
    }
    double sum = 0;
    for (int d = 0; d < D; ++d) O[(size_t)s * D + d] = 0;
    for (int t = 0; t < Skv; ++t) {
      if (causal && t > s + kv_off) continue;
      double d = 0;
      for (int k = 0; k < D; ++k)
        d += (double)Q[(size_t)s * D + k] * (double)K[(size_t)t * D + k];
      const double w = std::exp(d * scale - mx);
      sum += w;
      for (int k = 0; k < D; ++k) O[(size_t)s * D + k] += (float)(w * V[(size_t)t * D + k]);
    }
    for (int d = 0; d < D; ++d) O[(size_t)s * D + d] = (float)(O[(size_t)s * D + d] / sum);
  }
}

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
  std::mt19937 rng(0xA77E0u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const struct {
    int S, Skv, D, causal, kv_off;
    float scale;
  } cases[] = {
      {4, 4, 8, 0, 0, 0.0f},     // scale по умолчанию
      {8, 8, 16, 1, 0, 0.125f},  // causal
      {4, 8, 8, 1, 4, 0.0f},     // causal + сдвиг (decode)
      {2, 130, 32, 0, 0, 0.0f},  // Skv > 2*Bt: пересчёт m/l
      {16, 64, 64, 0, 0, 0.02f},
  };
  for (const auto &cs : cases) {
    std::vector<float> Q((size_t)cs.S * cs.D), K((size_t)cs.Skv * cs.D),
        V((size_t)cs.Skv * cs.D);
    // Масштаб входов 0.5, чтобы scores не улетали в насыщение exp.
    for (auto &v : Q) v = 0.5f * dist(rng);
    for (auto &v : K) v = 0.5f * dist(rng);
    for (auto &v : V) v = dist(rng);
    std::vector<float> got((size_t)cs.S * cs.D, 0), want(got.size(), 0);
    rouge_attention_f32(cs.S, cs.Skv, cs.D, Q.data(), K.data(), V.data(),
                        got.data(), cs.scale, cs.causal, cs.kv_off);
    ref_attn(cs.S, cs.Skv, cs.D, Q, K, V, want, cs.scale, cs.causal, cs.kv_off);
    const double e = rel_err(got, want);
    if (e > 1e-5) {
      std::printf("attn S=%d Skv=%d D=%d causal=%d kv=%d: rel %g\n", cs.S,
                  cs.Skv, cs.D, cs.causal, cs.kv_off, e);
      ++failures;
    }
  }
  // NULL/нули — без падения.
  rouge_attention_f32(0, 4, 8, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);

  if (failures == 0) std::printf("PASS: attention (5 cases vs reference)\n");
  return failures ? 1 : 0;
}
