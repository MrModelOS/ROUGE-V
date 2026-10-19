// AI-часть Wave 2: INT8-квантование, целочисленный GEMM, GQA-внимание с
// FP8-KV и трансформерный блок.
//
// Эталоны — намеренно примитивные (другой алгоритм), а не вызовы тех же
// функций: квантование сверяется с «идеальным» значением, GEMM — с целочисленным
// произведением в long double, внимание — с наивным softmax в double,
// трансформерный блок — с полностью наивной реализацией того же блока.

#include <algorithm>
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

// ---------------------------------------------------------------------------
// Наивный softmax-внимание (double) — эталон для GQA.
// ---------------------------------------------------------------------------
static void ref_attn_gqa(int S, int Skv, int Dh, int Hq, int Hkv,
                          const std::vector<float> &Q, const std::vector<float> &K,
                          const std::vector<float> &V, std::vector<float> &O,
                          int causal, int window, int kv_off) {
  const int group = Hq / Hkv;
  const double sc = 1.0 / std::sqrt((double)Dh);
  for (int hq = 0; hq < Hq; ++hq) {
    const int hk = hq / group;
    for (int s = 0; s < S; ++s) {
      int lo = 0, hi = Skv;
      if (causal) hi = std::min(Skv, s + kv_off + 1);
      if (window > 0) lo = std::max(0, s + kv_off - window + 1);
      std::vector<double> sc_scores;
      double mx = -1e300;
      for (int t = lo; t < hi; ++t) {
        const float *q = Q.data() + (size_t)hq * S * Dh + (size_t)s * Dh;
        const float *k = K.data() + (size_t)hk * Skv * Dh + (size_t)t * Dh;
        double dot = 0;
        for (int d = 0; d < Dh; ++d) dot += (double)q[d] * (double)k[d];
        sc_scores.push_back(dot * sc);
        mx = std::max(mx, sc_scores.back());
      }
      double sum = 0;
      for (double s2 : sc_scores) sum += std::exp(s2 - mx);
      for (int d = 0; d < Dh; ++d) {
        double acc = 0;
        int idx = 0;
        for (int t = lo; t < hi; ++t, ++idx) {
          const float *v = V.data() + (size_t)hk * Skv * Dh + (size_t)t * Dh;
          acc += (std::exp(sc_scores[idx] - mx) / sum) * (double)v[d];
        }
        O[(size_t)hq * S * Dh + (size_t)s * Dh + d] = (float)acc;
      }
    }
  }
}

int main() {
  std::mt19937 rng(0xA1u);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  // --- 1. INT8 per-tensor: roundtrip с точностью шага квантования ---
  {
    const int n = 1000;
    std::vector<float> x(n);
    for (auto &v : x) v = dist(rng);
    float scale = 0;
    std::vector<int8_t> q(n);
    EXPECT_TRUE(rouge_quantize_i8_tensor(x.data(), n, &scale, q.data()) == 1);
    EXPECT_TRUE(scale > 0.0f);
    std::vector<float> back(n);
    rouge_dequantize_i8_tensor(q.data(), n, scale, back.data());
    double worst = 0;
    for (int i = 0; i < n; ++i)
      worst = std::max(worst, (double)std::fabs(x[i] - back[i]));
    // Шаг квантования = scale; ошибка не больше половины шага (+eps).
    EXPECT_TRUE(worst <= scale * 0.5 + 1e-6);
    // Значения вне диапазона насыщаются: самый большой |x| -> 127.
    float s2 = 0;
    std::vector<int8_t> q2(1);
    std::vector<float> big = {1000.0f};
    rouge_quantize_i8_tensor(big.data(), 1, &s2, q2.data());
    EXPECT_TRUE(q2[0] == 127);
    std::vector<float> nbig = {-1000.0f};
    rouge_quantize_i8_tensor(nbig.data(), 1, &s2, q2.data());
    EXPECT_TRUE(q2[0] == -127);
  }

  // --- 2. INT8 per-channel: у каждой строки свой масштаб ---
  {
    const int rows = 8, cols = 64;
    std::vector<float> x((size_t)rows * cols);
    for (auto &v : x) v = dist(rng);
    x[0] = 100.0f;   // строка 0 — большая амплитуда
    x[1] = 0.001f;  // и крошечная в той же строке
    std::vector<float> scales(rows);
    std::vector<int8_t> q(x.size());
    EXPECT_TRUE(rouge_quantize_i8_channel(x.data(), rows, cols, scales.data(),
                                          q.data()) == 1);
    std::vector<float> back(x.size());
    rouge_dequantize_i8_channel(q.data(), rows, cols, scales.data(),
                                back.data());
    for (int r = 0; r < rows; ++r) {
      double amax = 0;
      for (int c = 0; c < cols; ++c)
        amax = std::max(amax, (double)std::fabs(x[(size_t)r * cols + c]));
      const double step = amax / 127.0;
      for (int c = 0; c < cols; ++c) {
        const double d = std::fabs(x[(size_t)r * cols + c] - back[(size_t)r * cols + c]);
        EXPECT_TRUE(d <= step * 0.5 + 1e-6);
      }
    }
  }

  // --- 3. INT8 per-group (асимметричный): диапазон группы покрыт точно ---
  // Граница ошибки — ШАГ, а не полушаг: нулевая точка целочисленная, поэтому
  // представляемый интервал сдвинут относительно [min,max] максимум на
  // полшага, и на границах к этому добавляется полушаг округления.
  // Такая же граница, как у асимметричного квантования в AWQ/GPTQ/TensorRT.
  {
    const int rows = 4, cols = 64, groups = 8;
    std::vector<float> x((size_t)rows * cols);
    for (auto &v : x) v = dist(rng);
    std::vector<float> scales(rows * groups);
    std::vector<uint8_t> zeros(rows * groups), q(x.size());
    EXPECT_TRUE(rouge_quantize_i8_group(x.data(), rows, cols, groups,
                                        scales.data(), zeros.data(),
                                        q.data()) == 1);
    std::vector<float> back(x.size());
    rouge_dequantize_i8_group(q.data(), rows, cols, groups, scales.data(),
                               zeros.data(), back.data());
    const int gsz = cols / groups;
    for (int r = 0; r < rows; ++r)
      for (int g = 0; g < groups; ++g) {
        double lo = 1e300, hi = -1e300;
        for (int i = 0; i < gsz; ++i) {
          const double v = x[(size_t)r * cols + g * gsz + i];
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
        // Шаг берём из СОХРАНЁННОГО масштаба (он float, поэтому сравнение с
        // double-эталоном даёт расхождение порядка 1e-7 от диапазона).
        const double step = (double)scales[r * groups + g];
        for (int i = 0; i < gsz; ++i) {
          const double d = std::fabs(x[(size_t)r * cols + g * gsz + i] -
                                     back[(size_t)r * cols + g * gsz + i]);
          EXPECT_TRUE(d <= step * (1.0 + 1e-5) + 1e-6);
        }
      }
  }

  // --- 4. Целочисленный GEMM: результаТ ТОЧНЫЙ (int32) против long double ---
  {
    const int M = 7, N = 5, K = 11;
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    float sa = 0, sb = 0;
    std::vector<int8_t> qa(A.size()), qb(B.size());
    rouge_quantize_i8_tensor(A.data(), (int)A.size(), &sa, qa.data());
    rouge_quantize_i8_tensor(B.data(), (int)B.size(), &sb, qb.data());
    std::vector<float> C((size_t)M * N, 0.0f);
    rouge_gemm_i8(M, N, K, qa.data(), qb.data(), &sa, 0, &sb, 0, C.data());
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        long long acc = 0;
        for (int k = 0; k < K; ++k)
          acc += (long long)qa[(size_t)m * K + k] * (long long)qb[(size_t)k * N + n];
        const long long want = acc;
        const long long got = (long long)std::llround(
            (double)C[(size_t)m * N + n] / ((double)sa * (double)sb));
        EXPECT_TRUE(want == got);
      }
  }

  // --- 5. GQA-внимание против наивного эталона ---
  {
    const int S = 12, Skv = 12, Dh = 8, Hq = 4, Hkv = 2;
    std::vector<float> Q((size_t)Hq * S * Dh), K((size_t)Hkv * Skv * Dh),
        V((size_t)Hkv * Skv * Dh), O((size_t)Hq * S * Dh);
    for (auto &v : Q) v = dist(rng);
    for (auto &v : K) v = dist(rng);
    for (auto &v : V) v = dist(rng);
    for (int causal : {0, 1})
      for (int window : {0, 4}) {
        std::vector<float> ref((size_t)Hq * S * Dh);
        ref_attn_gqa(S, Skv, Dh, Hq, Hkv, Q, K, V, ref, causal, window, 0);
        rouge_attention_gqa(S, Skv, Dh, Hq, Hkv, Q.data(), K.data(), V.data(),
                            O.data(), 0.0f, causal, window, 0);
        double worst = 0;
        for (size_t i = 0; i < O.size(); ++i)
          worst = std::max(worst, (double)std::fabs(O[i] - ref[i]));
        EXPECT_TRUE(worst < 1e-5);
      }
  }

  // --- 6. FP8-KV: близко к f32-вниманию (ошибка квантования кэша) ---
  {
    const int S = 10, Skv = 10, Dh = 16, Hq = 4, Hkv = 4;
    std::vector<float> Q((size_t)Hq * S * Dh), K((size_t)Hkv * Skv * Dh),
        V((size_t)Hkv * Skv * Dh);
    for (auto &v : Q) v = dist(rng);
    for (auto &v : K) v = dist(rng);
    for (auto &v : V) v = dist(rng);
    // Квантуем K/V в e4m3 с масштабом на голову (у K и V свой масштаб).
    std::vector<uint8_t> Kq((size_t)Hkv * Skv * Dh), Vq(Kq.size());
    std::vector<float> sk(Hkv), sv(Hkv);
    for (int h = 0; h < Hkv; ++h) {
      float amaxk = 0, amaxv = 0;
      for (int t = 0; t < Skv; ++t)
        for (int d = 0; d < Dh; ++d) {
          amaxk = std::max(amaxk, std::fabs(K[(size_t)h * Skv * Dh + (size_t)t * Dh + d]));
          amaxv = std::max(amaxv, std::fabs(V[(size_t)h * Skv * Dh + (size_t)t * Dh + d]));
        }
      sk[h] = amaxk > 0 ? amaxk / 448.0f : 1.0f;
      sv[h] = amaxv > 0 ? amaxv / 448.0f : 1.0f;
      for (int t = 0; t < Skv; ++t)
        for (int d = 0; d < Dh; ++d) {
          const size_t i = (size_t)h * Skv * Dh + (size_t)t * Dh + d;
          Kq[i] = rouge_quantize_e4m3(K[i] / sk[h]);
          Vq[i] = rouge_quantize_e4m3(V[i] / sv[h]);
        }
    }
    std::vector<float> O((size_t)Hq * S * Dh), ref((size_t)Hq * S * Dh);
    ref_attn_gqa(S, Skv, Dh, Hq, Hkv, Q, K, V, ref, 1, 0, 0);
    rouge_attention_gqa_fp8kv(S, Skv, Dh, Hq, Hkv, Q.data(), Kq.data(),
                              Vq.data(), sk.data(), sv.data(), O.data(), 0.0f, 1,
                              0, 0);
    // Ошибка квантования KV в e4m3: 3 бита мантиссы -> полу-ULP 2^-4 = 6.25%
    // от модуля значения. Проверяем относительно масштаба самого ответа.
    double se = 0, sr = 0;
    for (size_t i = 0; i < O.size(); ++i) {
      const double e = (double)O[i] - (double)ref[i];
      se += e * e;
      sr += (double)ref[i] * (double)ref[i];
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    EXPECT_TRUE(rel < 0.10);
  }

  // --- 7. Трансформерный блок против наивной реализации ---
  {
    const int S = 6, D = 16, Hq = 2, Hkv = 1, Dq = 8, Dk = 8, Dff = 24;
    std::vector<float> x((size_t)S * D), wrms1(D), wrms2(D);
    // Раскладка весов как у nn.Linear: [вход][выход].
    std::vector<float> wq((size_t)D * Hq * Dq), wk((size_t)D * Hkv * Dk),
        wv((size_t)D * Hkv * Dk), wo((size_t)Hq * Dq * D);
    std::vector<float> wgate((size_t)D * Dff), wup((size_t)D * Dff),
        wdown((size_t)Dff * D);
    for (auto &v : x) v = dist(rng);
    for (auto &v : wrms1) v = 0.5f + 0.1f * std::fabs(v);
    for (auto &v : wrms2) v = 0.5f + 0.1f * std::fabs(v);
    for (auto &v : wq) v = dist(rng) * 0.1f;
    for (auto &v : wk) v = dist(rng) * 0.1f;
    for (auto &v : wv) v = dist(rng) * 0.1f;
    for (auto &v : wo) v = dist(rng) * 0.1f;
    for (auto &v : wgate) v = dist(rng) * 0.1f;
    for (auto &v : wup) v = dist(rng) * 0.1f;
    for (auto &v : wdown) v = dist(rng) * 0.1f;
    std::vector<float> out((size_t)S * D);
    rouge_transformer_block(S, D, Hq, Hkv, Dq, Dk, x.data(), wrms1.data(),
                             wq.data(), wk.data(), wv.data(), wo.data(),
                             wrms2.data(), wgate.data(), wup.data(),
                             wdown.data(), Dff, 10000.0f, 1e-5f, out.data());
    // Наивный эталон: тот же блок, но каждая операция «в лоб».
    std::vector<float> ref((size_t)S * D);
    {
      auto rms = [](const float *in, int n, const float *w, float eps,
                    float *o) {
        double ss = 0;
        for (int i = 0; i < n; ++i) ss += (double)in[i] * (double)in[i];
        const double inv = 1.0 / std::sqrt(ss / n + eps);
        for (int i = 0; i < n; ++i) o[i] = (float)((double)in[i] * inv) * w[i];
      };
      auto rope1 = [](float *v, int n, int pos, float theta) {
        const int half = n / 2;
        for (int i = 0; i < half; ++i) {
          const double freq = 1.0 / std::pow((double)theta, (2.0 * i) / (double)n);
          const double a = (double)pos * freq;
          const float c = (float)std::cos(a), s = (float)std::sin(a);
          const float x0 = v[i], x1 = v[i + half];
          v[i] = x0 * c - x1 * s;
          v[i + half] = x0 * s + x1 * c;
        }
      };
      std::vector<float> norm((size_t)S * D), q((size_t)Hq * S * Dq),
          k((size_t)Hkv * S * Dk), v((size_t)Hkv * S * Dk),
          attn((size_t)Hq * S * Dq), h1((size_t)S * D), norm2((size_t)S * D),
          gate((size_t)S * Dff), up((size_t)S * Dff);
      for (int s = 0; s < S; ++s)
        rms(x.data() + (size_t)s * D, D, wrms1.data(), 1e-5f,
            norm.data() + (size_t)s * D);
      for (int h = 0; h < Hq; ++h)
        for (int s = 0; s < S; ++s)
          for (int i = 0; i < Dq; ++i) {
            double acc = 0;
            for (int d = 0; d < D; ++d)
              acc += (double)norm[(size_t)s * D + d] *
                     (double)wq[(size_t)d * Hq * Dq + h * Dq + i];
            q[(size_t)h * S * Dq + (size_t)s * Dq + i] = (float)acc;
          }
      for (int h = 0; h < Hkv; ++h)
        for (int s = 0; s < S; ++s)
          for (int i = 0; i < Dk; ++i) {
            double ak = 0, av = 0;
            for (int d = 0; d < D; ++d) {
              const double xv = norm[(size_t)s * D + d];
              ak += xv * (double)wk[(size_t)d * Hkv * Dk + h * Dk + i];
              av += xv * (double)wv[(size_t)d * Hkv * Dk + h * Dk + i];
            }
            k[(size_t)h * S * Dk + (size_t)s * Dk + i] = (float)ak;
            v[(size_t)h * S * Dk + (size_t)s * Dk + i] = (float)av;
          }
      for (int h = 0; h < Hq; ++h)
        for (int s = 0; s < S; ++s)
          rope1(q.data() + (size_t)h * S * Dq + (size_t)s * Dq, Dq, s, 10000.0f);
      for (int h = 0; h < Hkv; ++h)
        for (int s = 0; s < S; ++s)
          rope1(k.data() + (size_t)h * S * Dk + (size_t)s * Dk, Dk, s, 10000.0f);
      for (int hq = 0; hq < Hq; ++hq) {
        const int hk = hq / (Hq / Hkv);
        for (int s = 0; s < S; ++s) {
          std::vector<double> sc(0);
          double mx = -1e300;
          for (int t = 0; t <= s; ++t) {
            double dot = 0;
            for (int d = 0; d < Dq; ++d)
              dot += (double)q[(size_t)hq * S * Dq + (size_t)s * Dq + d] *
                     (double)k[(size_t)hk * S * Dk + (size_t)t * Dk + d];
            sc.push_back(dot / std::sqrt((double)Dq));
            mx = std::max(mx, sc.back());
          }
          double sum = 0;
          for (double x : sc) sum += std::exp(x - mx);
          for (int d = 0; d < Dq; ++d) {
            double acc = 0;
            for (size_t t = 0; t < sc.size(); ++t)
              acc += (std::exp(sc[t] - mx) / sum) *
                     (double)v[(size_t)hk * S * Dk + t * Dk + d];
            attn[(size_t)hq * S * Dq + (size_t)s * Dq + d] = (float)acc;
          }
        }
      }
      // Внимание лежит в [Hq][S][Dq], wo — в [Hq*Dq][D].
      for (int s = 0; s < S; ++s)
        for (int d = 0; d < D; ++d) {
          double acc = 0;
          for (int hq = 0; hq < Hq; ++hq)
            for (int dd = 0; dd < Dq; ++dd)
              acc += (double)attn[((size_t)hq * S + s) * Dq + dd] *
                     (double)wo[(size_t)(hq * Dq + dd) * D + d];
          h1[(size_t)s * D + d] = x[(size_t)s * D + d] + (float)acc;
        }
      for (int s = 0; s < S; ++s)
        rms(h1.data() + (size_t)s * D, D, wrms2.data(), 1e-5f,
            norm2.data() + (size_t)s * D);
      for (int s = 0; s < S; ++s)
        for (int i = 0; i < Dff; ++i) {
          double ag = 0, au = 0;
          for (int d = 0; d < D; ++d) {
            const double xv = norm2[(size_t)s * D + d];
            ag += xv * (double)wgate[(size_t)d * Dff + i];
            au += xv * (double)wup[(size_t)d * Dff + i];
          }
          gate[(size_t)s * Dff + i] = (float)ag;
          up[(size_t)s * Dff + i] = (float)au;
        }
      for (int s = 0; s < S; ++s)
        for (int i = 0; i < Dff; ++i) {
          const float g = gate[(size_t)s * Dff + i];
          gate[(size_t)s * Dff + i] = (g / (1.0f + std::exp(-g))) * up[(size_t)s * Dff + i];
        }
      for (int s = 0; s < S; ++s)
        for (int d = 0; d < D; ++d) {
          double acc = 0;
          for (int i = 0; i < Dff; ++i)
            acc += (double)gate[(size_t)s * Dff + i] *
                   (double)wdown[(size_t)i * D + d];
          ref[(size_t)s * D + d] = h1[(size_t)s * D + d] + (float)acc;
        }
    }
    double worst = 0;
    for (size_t i = 0; i < out.size(); ++i)
      worst = std::max(worst, (double)std::fabs(out[i] - ref[i]));
    // Блок считает проекции нашим f32-GEMM, эталон — в double, поэтому
    // расхождение — это разница точности накопления, а не ошибка алгоритма.
    std::printf("  трансформерный блок: расхождение с наивным эталоном %.2e\n",
                worst);
    EXPECT_TRUE(worst < 1e-4);
  }

  if (failures == 0) std::printf("PASS: int8 + gqa attention + transformer block\n");
  return failures ? 1 : 0;
}
