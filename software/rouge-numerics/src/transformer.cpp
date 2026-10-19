/*
 * Трансформерный блок (Wave 2, AI): RMSNorm -> RoPE -> GQA-внимание ->
 * выходная проекция -> SwiGLU-MLP, всё на наших же ядрах.
 *
 * Это не «модель», а блок — ровно та единица, которую нужно, чтобы
 * показать: наш путь считает настоящий трансформерный слой, а не «GEMM с
 * маской». Состав (pre-norm, как в LLaMA-подобных):
 *   h1 = x + Attn(RMSNorm(x) * RoPE)
 *   y  = h1 + MLP(RMSNorm(h1))
 *   MLP: down(silu(gate(x)) * up(x))
 *
 * Проекции считаются НАШИМ блочным GEMM с автотюнером (rouge_gemm_f32),
 * а не наивной тройной петлёй: наивный вариант давал 2.6 GFLOP/s, с GEMM
 * блок ускоряется в разы (замер в tests/bench_ai.cpp).
 *
 * Раскладка весов — как у nn.Linear: [вход][выход]:
 *   wq [D][Hq*Dq], wk/wv [D][Hkv*Dk], wo [Hq*Dq][D],
 *   wgate/wup [D][Dff], wdown [Dff][D].
 *
 * Все веса — f32, в тестах — против полностью наивной реализации
 * (тест проверяет совпадение, а не «примерно»).
 */

#include "rouge/numerics.h"

#include <cmath>
#include <cstddef>
#include <vector>

extern "C" {

// RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i].
void rouge_rmsnorm(const float *x, int n, const float *w, float eps,
                   float *out) {
  if (!x || !w || !out || n <= 0) return;
  double ss = 0.0;
  for (int i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
  const double inv = 1.0 / std::sqrt(ss / (double)n + (double)eps);
  for (int i = 0; i < n; ++i) out[i] = (float)((double)x[i] * inv) * w[i];
}

// RoPE (по двум половинам, как в llama.cpp): пары (i, i + n/2) поворачиваются
// на угол pos * inv_freq^(i / (n/2)). Применяется к head_dim.
void rouge_rope(float *x, int heads, int head_dim, int pos, float theta) {
  if (!x || heads <= 0 || head_dim <= 0) return;
  const int half = head_dim / 2;
  for (int h = 0; h < heads; ++h) {
    float *v = x + (size_t)h * head_dim;
    for (int i = 0; i < half; ++i) {
      const double freq =
          1.0 / std::pow((double)theta, (2.0 * i) / (double)head_dim);
      const double a = (double)pos * freq;
      const float c = (float)std::cos(a), s = (float)std::sin(a);
      const float x0 = v[i], x1 = v[i + half];
      v[i] = x0 * c - x1 * s;
      v[i + half] = x0 * s + x1 * c;
    }
  }
}

// SwiGLU: out = silu(gate) * up, silu(z) = z / (1 + e^-z).
void rouge_swiglu(const float *gate, const float *up, int n, float *out) {
  if (!gate || !up || !out || n <= 0) return;
  for (int i = 0; i < n; ++i) {
    const float g = gate[i];
    out[i] = (g / (1.0f + std::exp(-g))) * up[i];
  }
}

// Полный блок. Раскладка:
//   x        : [S][D]            — вход (S токенов, D признаков)
//   wrms1    : [D]               — веса RMSNorm перед вниманием
//   wq       : [D][Hq*Dq]        — проекция запроса
//   wk/wv    : [D][Hkv*Dk]       — проекции ключа и значения
//   wo       : [Hq*Dq][D]        — выходная проекция внимания
//   wrms2    : [D]               — RMSNorm перед MLP
//   wgate/wup: [D][Dff]          — SwiGLU
//   wdown    : [Dff][D]
void rouge_transformer_block(int S, int D, int Hq, int Hkv, int Dq, int Dk,
                              const float *x, const float *wrms1,
                              const float *wq, const float *wk, const float *wv,
                              const float *wo, const float *wrms2,
                              const float *wgate, const float *wup,
                              const float *wdown, int Dff, float rope_theta,
                              float rms_eps, float *out) {
  if (!x || !out || S <= 0 || D <= 0) return;
  // Внимание у нас с общим размером головы: Dk должен совпадать с Dq.
  if (Dk != Dq || Hq <= 0 || Hkv <= 0 || Hq % Hkv != 0) return;
  if (Dff <= 0) return;

  std::vector<float> norm((size_t)S * D);
  std::vector<float> qflat((size_t)S * Hq * Dq);  // [S][Hq*Dq]
  std::vector<float> kflat((size_t)S * Hkv * Dq);
  std::vector<float> vflat((size_t)S * Hkv * Dq);
  std::vector<float> attn((size_t)Hq * S * Dq);
  std::vector<float> attnflat((size_t)S * Hq * Dq);
  std::vector<float> proj((size_t)S * D);
  std::vector<float> h1((size_t)S * D);
  std::vector<float> norm2((size_t)S * D);
  std::vector<float> gate((size_t)S * Dff), up((size_t)S * Dff);
  std::vector<float> down((size_t)S * D);

  // 1. RMSNorm.
  for (int s = 0; s < S; ++s)
    rouge_rmsnorm(x + (size_t)s * D, D, wrms1, rms_eps,
                   norm.data() + (size_t)s * D);

  // 2. Проекции Q/K/V — нашим блочным GEMM.
  rouge_gemm_f32(S, Hq * Dq, D, norm.data(), wq, qflat.data(), nullptr);
  rouge_gemm_f32(S, Hkv * Dq, D, norm.data(), wk, kflat.data(), nullptr);
  rouge_gemm_f32(S, Hkv * Dq, D, norm.data(), wv, vflat.data(), nullptr);

  // 3. Раскладка [S][Hq*Dq] -> [Hq][S][Dq] и RoPE.
  std::vector<float> q((size_t)Hq * S * Dq);
  std::vector<float> k((size_t)Hkv * S * Dq);
  std::vector<float> v((size_t)Hkv * S * Dq);
  for (int hq = 0; hq < Hq; ++hq)
    for (int s = 0; s < S; ++s)
      for (int d = 0; d < Dq; ++d)
        q[((size_t)hq * S + s) * Dq + d] = qflat[((size_t)s * Hq + hq) * Dq + d];
  for (int hk = 0; hk < Hkv; ++hk)
    for (int s = 0; s < S; ++s)
      for (int d = 0; d < Dq; ++d) {
        k[((size_t)hk * S + s) * Dq + d] = kflat[((size_t)s * Hkv + hk) * Dq + d];
        v[((size_t)hk * S + s) * Dq + d] = vflat[((size_t)s * Hkv + hk) * Dq + d];
      }
  for (int h = 0; h < Hq; ++h)
    for (int s = 0; s < S; ++s)
      rouge_rope(q.data() + ((size_t)h * S + s) * Dq, 1, Dq, s, rope_theta);
  for (int h = 0; h < Hkv; ++h)
    for (int s = 0; s < S; ++s)
      rouge_rope(k.data() + ((size_t)h * S + s) * Dq, 1, Dq, s, rope_theta);

  // 4. Внимание (causal, без окна).
  rouge_attention_gqa(S, S, Dq, Hq, Hkv, q.data(), k.data(), v.data(),
                      attn.data(), 0.0f, 1, 0, 0);

  // 5. Обратная раскладка и выходная проекция, затем residual.
  for (int hq = 0; hq < Hq; ++hq)
    for (int s = 0; s < S; ++s)
      for (int d = 0; d < Dq; ++d)
        attnflat[((size_t)s * Hq + hq) * Dq + d] = attn[((size_t)hq * S + s) * Dq + d];
  rouge_gemm_f32(S, D, Hq * Dq, attnflat.data(), wo, proj.data(), nullptr);
  for (size_t i = 0; i < (size_t)S * D; ++i)
    h1[i] = x[i] + proj[i];

  // 6. MLP: RMSNorm -> SwiGLU -> down -> residual.
  for (int s = 0; s < S; ++s)
    rouge_rmsnorm(h1.data() + (size_t)s * D, D, wrms2, rms_eps,
                   norm2.data() + (size_t)s * D);
  rouge_gemm_f32(S, Dff, D, norm2.data(), wgate, gate.data(), nullptr);
  rouge_gemm_f32(S, Dff, D, norm2.data(), wup, up.data(), nullptr);
  for (size_t i = 0; i < (size_t)S * Dff; ++i) {
    const float g = gate[i];
    gate[i] = (g / (1.0f + std::exp(-g))) * up[i];
  }
  rouge_gemm_f32(S, D, Dff, gate.data(), wdown, down.data(), nullptr);
  for (size_t i = 0; i < (size_t)S * D; ++i) out[i] = h1[i] + down[i];
}

}  // extern "C"
