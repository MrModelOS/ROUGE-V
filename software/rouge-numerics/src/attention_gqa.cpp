/*
 * Внимание для инференса: GQA/MQA, скользящее окно и FP8-KV (Wave 2, AI).
 *
 * Отличие от attention.cpp: там одна «голова внимания» на все головы с
 * общим K/V (MHA). Здесь — то, что нужно современному инференсу:
 *   * GQA/MQA: у голов запроса Hq голов K/V может быть меньше (Hq % Hkv == 0),
 *     каждая группа голов запроса делит одну голову K/V;
 *   * скользящее окно: у attention, только t in [s-window+1, s] (длинный
 *     контекст с локальной рецепцией);
 *   * FP8-KV: кэш K/V хранится в e4m3 (собственный формат проекта) с
 *     масштабом на голову, при чтении деквантуется на лету.
 *
 * Раскладка: Q — [Hq][S][Dh], K/V — [Hkv][Skv][Dh], O — [Hq][S][Dh].
 * Всё row-major, без внешних зависимостей.
 *
 * Устройство (важно для скорости):
 *   * блок по t (Bt=64) с онлайн-нормализацией: полная матрица S×Skv
 *     не материализуется, накопление идёт через running max/sum;
 *   * блок K/V копируется в локальный буфер ОДИН раз и используется всеми
 *     головами запроса группы — иначе K/V читался бы заново для каждой
 *     головы, и GQA был бы медленнее MHA в несколько раз (проверено
 *     замером: до переработки — 26x медленнее при 4x меньшем объёме K/V);
 *   * FP8-KV деквантуется в тот же локальный буфер.
 *
 * Точность: накопители m/l — float (как в attention.cpp), скалярные
 * произведения считаются в double. Эталон в тестах — наивный softmax в
 * double; расхождение проверяется тестом.
 */

#include "rouge/numerics.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

// Масштаб softmax: если scale <= 0 — 1/sqrt(Dh).
inline float softmax_scale(float scale, int dh) {
  return scale > 0.0f ? scale : 1.0f / std::sqrt((float)dh);
}

constexpr int kBlock = 64;

}  // namespace

extern "C" {

// Внимание с GQA и скользящим окном.
//   window <= 0 — окно не ограничено (только causal, если causal != 0).
//   kv_off      — сдвиг позиций K/V относительно Q (decode с префиксом).
void rouge_attention_gqa(int S, int Skv, int Dh, int Hq, int Hkv, const float *Q,
                         const float *K, const float *V, float *O, float scale,
                         int causal, int window, int kv_off) {
  if (!Q || !K || !V || !O) return;
  if (S <= 0 || Skv <= 0 || Dh <= 0 || Hq <= 0 || Hkv <= 0) return;
  if (Hq % Hkv != 0) return;  // GQA требует кратности
  const int group = Hq / Hkv;
  const float sc = softmax_scale(scale, Dh);

  // Аккумуляторы: для каждой головы запроса и токена — running max, сумма
  // exp и накопленный выход (D значений).
  const size_t nq = (size_t)Hq * S;
  std::vector<float> m(nq, -INFINITY);
  std::vector<float> l(nq, 0.0f);
  std::vector<float> acc((size_t)Hq * S * Dh, 0.0f);

  std::vector<float> kbuf((size_t)kBlock * Dh);
  std::vector<float> vbuf((size_t)kBlock * Dh);
  std::vector<float> scores(kBlock);

  for (int hk = 0; hk < Hkv; ++hk) {
    const int hq0 = hk * group;
    const float *Kh = K + (size_t)hk * Skv * Dh;
    const float *Vh = V + (size_t)hk * Skv * Dh;
    for (int hq = hq0; hq < hq0 + group; ++hq) {
      const size_t base = (size_t)hq * S;
      for (int s = 0; s < S; ++s)
        for (int d = 0; d < Dh; ++d) acc[(base + (size_t)s) * Dh + d] = 0.0f;
    }

    for (int t0 = 0; t0 < Skv; t0 += kBlock) {
      const int t1 = t0 + kBlock < Skv ? t0 + kBlock : Skv;
      const int bn = t1 - t0;
      // Копируем блок K/V в локальный буфер: он читается всеми головами
      // запроса группы, поэтому копирование окупается.
      for (int t = 0; t < bn; ++t) {
        const float *k = Kh + (size_t)(t0 + t) * Dh;
        const float *v = Vh + (size_t)(t0 + t) * Dh;
        float *kd = kbuf.data() + (size_t)t * Dh;
        float *vd = vbuf.data() + (size_t)t * Dh;
        for (int d = 0; d < Dh; ++d) {
          kd[d] = k[d];
          vd[d] = v[d];
        }
      }

      for (int hq = hq0; hq < hq0 + group; ++hq) {
        const float *Qh = Q + (size_t)hq * S * Dh;
        const size_t base = (size_t)hq * S;
        for (int s = 0; s < S; ++s) {
          // Диапазон по t: causal + окно.
          int t_lo = 0, t_hi = Skv;
          if (causal) {
            t_hi = s + kv_off + 1;
            if (t_hi > Skv) t_hi = Skv;
          }
          if (window > 0) {
            const int lo = s + kv_off - window + 1;
            if (lo > t_lo) t_lo = lo;
          }
          if (t_lo < 0) t_lo = 0;
          if (t_hi <= t_lo) continue;

          const float *q = Qh + (size_t)s * Dh;
          float *o = acc.data() + (base + (size_t)s) * Dh;
          float &mx = m[base + (size_t)s];
          float &sum = l[base + (size_t)s];

          // 1. Scores блока.
          float block_max = -INFINITY;
          const int b_lo = t_lo > t0 ? t_lo : t0;
          const int b_hi = t_hi < t1 ? t_hi : t1;
          for (int t = b_lo; t < b_hi; ++t) {
            const float *k = kbuf.data() + (size_t)(t - t0) * Dh;
            double dot = 0.0;
            for (int d = 0; d < Dh; ++d) dot += (double)q[d] * (double)k[d];
            const float v = (float)(dot * sc);
            scores[(size_t)(t - t0)] = v;
            if (v > block_max) block_max = v;
          }
          if (block_max == -INFINITY) continue;

          // 2. Пересчёт максимума и коэффициента забывания.
          const float m_new = mx > block_max ? mx : block_max;
          const float alpha = (mx == -INFINITY) ? 0.0f : std::exp(mx - m_new);
          float block_sum = 0.0f;
          for (int t = b_lo; t < b_hi; ++t)
            block_sum += std::exp(scores[(size_t)(t - t0)] - m_new);
          const float l_new = sum * alpha + block_sum;

          // 3. Пересчёт выхода.
          const float o_scale = (l_new == 0.0f) ? 0.0f : (sum * alpha / l_new);
          const float b_scale = (l_new == 0.0f) ? 0.0f : (1.0f / l_new);
          for (int d = 0; d < Dh; ++d) o[d] *= o_scale;
          for (int t = b_lo; t < b_hi; ++t) {
            const float w = std::exp(scores[(size_t)(t - t0)] - m_new) * b_scale;
            const float *v = vbuf.data() + (size_t)(t - t0) * Dh;
            for (int d = 0; d < Dh; ++d) o[d] += w * v[d];
          }
          mx = m_new;
          sum = l_new;
        }
      }
    }
  }

  for (size_t i = 0; i < nq * (size_t)Dh; ++i) O[i] = acc[i];
}

// FP8-KV: кэш K/V заранее квантован в e4m3 с масштабом на голову.
//   Kq/Vq: [Hkv][Skv][Dh] байт e4m3, scales_k/scales_v: [Hkv] значений.
// Деквантование на лету: значение = e4m3_декод(байт) * scale.
void rouge_attention_gqa_fp8kv(int S, int Skv, int Dh, int Hq, int Hkv,
                               const float *Q, const uint8_t *Kq,
                               const uint8_t *Vq, const float *scales_k,
                               const float *scales_v, float *O, float scale,
                               int causal, int window, int kv_off) {
  if (!Q || !Kq || !Vq || !scales_k || !scales_v || !O) return;
  if (S <= 0 || Skv <= 0 || Dh <= 0 || Hq <= 0 || Hkv <= 0) return;
  if (Hq % Hkv != 0) return;
  const int group = Hq / Hkv;
  const float sc = softmax_scale(scale, Dh);

  const size_t nq = (size_t)Hq * S;
  std::vector<float> m(nq, -INFINITY);
  std::vector<float> l(nq, 0.0f);
  std::vector<float> acc((size_t)Hq * S * Dh, 0.0f);

  std::vector<float> kbuf((size_t)kBlock * Dh);
  std::vector<float> vbuf((size_t)kBlock * Dh);
  std::vector<float> scores(kBlock);

  for (int hk = 0; hk < Hkv; ++hk) {
    const int hq0 = hk * group;
    const uint8_t *Kh = Kq + (size_t)hk * Skv * Dh;
    const uint8_t *Vh = Vq + (size_t)hk * Skv * Dh;
    const float sk = scales_k[hk], sv = scales_v[hk];
    for (int hq = hq0; hq < hq0 + group; ++hq) {
      const size_t base = (size_t)hq * S;
      for (int s = 0; s < S; ++s)
        for (int d = 0; d < Dh; ++d) acc[(base + (size_t)s) * Dh + d] = 0.0f;
    }

    for (int t0 = 0; t0 < Skv; t0 += kBlock) {
      const int t1 = t0 + kBlock < Skv ? t0 + kBlock : Skv;
      const int bn = t1 - t0;
      // Деквантование блока K/V в локальный буфер — один раз на блок.
      for (int t = 0; t < bn; ++t) {
        const uint8_t *kb = Kh + (size_t)(t0 + t) * Dh;
        const uint8_t *vb = Vh + (size_t)(t0 + t) * Dh;
        float *kd = kbuf.data() + (size_t)t * Dh;
        float *vd = vbuf.data() + (size_t)t * Dh;
        for (int d = 0; d < Dh; ++d) {
          kd[d] = rouge_dequantize_e4m3(kb[d]) * sk;
          vd[d] = rouge_dequantize_e4m3(vb[d]) * sv;
        }
      }

      for (int hq = hq0; hq < hq0 + group; ++hq) {
        const float *Qh = Q + (size_t)hq * S * Dh;
        const size_t base = (size_t)hq * S;
        for (int s = 0; s < S; ++s) {
          int t_lo = 0, t_hi = Skv;
          if (causal) {
            t_hi = s + kv_off + 1;
            if (t_hi > Skv) t_hi = Skv;
          }
          if (window > 0) {
            const int lo = s + kv_off - window + 1;
            if (lo > t_lo) t_lo = lo;
          }
          if (t_lo < 0) t_lo = 0;
          if (t_hi <= t_lo) continue;

          const float *q = Qh + (size_t)s * Dh;
          float *o = acc.data() + (base + (size_t)s) * Dh;
          float &mx = m[base + (size_t)s];
          float &sum = l[base + (size_t)s];

          float block_max = -INFINITY;
          const int b_lo = t_lo > t0 ? t_lo : t0;
          const int b_hi = t_hi < t1 ? t_hi : t1;
          for (int t = b_lo; t < b_hi; ++t) {
            const float *k = kbuf.data() + (size_t)(t - t0) * Dh;
            double dot = 0.0;
            for (int d = 0; d < Dh; ++d) dot += (double)q[d] * (double)k[d];
            const float v = (float)(dot * sc);
            scores[(size_t)(t - t0)] = v;
            if (v > block_max) block_max = v;
          }
          if (block_max == -INFINITY) continue;

          const float m_new = mx > block_max ? mx : block_max;
          const float alpha = (mx == -INFINITY) ? 0.0f : std::exp(mx - m_new);
          float block_sum = 0.0f;
          for (int t = b_lo; t < b_hi; ++t)
            block_sum += std::exp(scores[(size_t)(t - t0)] - m_new);
          const float l_new = sum * alpha + block_sum;

          const float o_scale = (l_new == 0.0f) ? 0.0f : (sum * alpha / l_new);
          const float b_scale = (l_new == 0.0f) ? 0.0f : (1.0f / l_new);
          for (int d = 0; d < Dh; ++d) o[d] *= o_scale;
          for (int t = b_lo; t < b_hi; ++t) {
            const float w = std::exp(scores[(size_t)(t - t0)] - m_new) * b_scale;
            const float *v = vbuf.data() + (size_t)(t - t0) * Dh;
            for (int d = 0; d < Dh; ++d) o[d] += w * v[d];
          }
          mx = m_new;
          sum = l_new;
        }
      }
    }
  }

  for (size_t i = 0; i < nq * (size_t)Dh; ++i) O[i] = acc[i];
}

}  // extern "C"
