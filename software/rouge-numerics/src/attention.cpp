/*
 * Attention Wave 2: блочный softmax-attention с онлайн-нормализацией
 * (FlashAttention-подобный приём, но свой код, не копия алгоритма из статей
 * дословно: идея tiling + running max/sum — общеизвестный приём численной
 * устойчивости softmax).
 *
 *   O[s][d] = sum_t softmax_t( Q[s]·K[t]/scale ) * V[t][d]
 *
 * Полная матрица S×Skv никогда не материализуется: проход по блокам K/V
 * (Bt=64), текущие m (max) и l (sumexp) с пересчётом O при росте m.
 * Сложность O(S*Skv*D) по операциям, память O(S*D + Bt*D).
 * causal=0/1: при causal маскируем t > s_global + kv_off (декодер).
 * kv_off: сдвиг K/V-позиций относительно Q (для continuing decode); 0 обычно.
 * scale <= 0 -> 1/sqrt(D).
 *
 * Точность: накопители m/l — double внутри блока не нужен, float достаточно
 * при пересчёте через expf; эталон — наивный двухпроходной softmax в double.
 */

#include "rouge/numerics.h"

#include <cmath>
#include <cstddef>

extern "C" {

void rouge_attention_f32(int S, int Skv, int D, const float *Q, const float *K,
                         const float *V, float *O, float scale, int causal,
                         int kv_off) {
  if (!Q || !K || !V || !O) return;
  if (S <= 0 || Skv <= 0 || D <= 0) return;
  if (scale <= 0.0f) scale = 1.0f / std::sqrt((float)D);

  constexpr int Bt = 64;
  // Временный буфер scores одного блока: на стек не лезет при Bt=64 -> S? нет,
  // блок по t фиксирован: Bt скаляров.
  for (int s = 0; s < S; ++s) {
    const float *q = Q + (size_t)s * D;
    float *o = O + (size_t)s * D;
    for (int d = 0; d < D; ++d) o[d] = 0.0f;
    float m = -INFINITY;  // текущий максимум
    float l = 0.0f;       // текущая сумма exp
    for (int t0 = 0; t0 < Skv; t0 += Bt) {
      int t1 = t0 + Bt < Skv ? t0 + Bt : Skv;
      // 1. scores блока.
      float block_max = -INFINITY;
      // scores храним компактно на стеке через фиксированный массив.
      float sc[64];
      for (int t = t0; t < t1; ++t) {
        float v;
        if (causal && t > s + kv_off) {
          v = -INFINITY;
        } else {
          const float *k = K + (size_t)t * D;
          double dot = 0.0;
          for (int d = 0; d < D; ++d) dot += (double)q[d] * (double)k[d];
          v = (float)(dot * scale);
        }
        sc[t - t0] = v;
        if (v > block_max) block_max = v;
      }
      if (block_max == -INFINITY) continue;  // вся строка замаскирована
      // 2. пересчёт: новый общий максимум и коэффициент забывания.
      const float m_new = m > block_max ? m : block_max;
      const float alpha = (m == -INFINITY) ? 0.0f : std::exp(m - m_new);
      float block_sum = 0.0f;
      for (int t = t0; t < t1; ++t) {
        if (sc[t - t0] == -INFINITY) continue;
        block_sum += std::exp(sc[t - t0] - m_new);
      }
      const float l_new = l * alpha + block_sum;
      // 3. O = O*alpha*(l/l_new)... точнее: O_old * (l*alpha/l_new) + block/l_new.
      const float o_scale = (l_new == 0.0f) ? 0.0f : (l * alpha / l_new);
      const float b_scale = (l_new == 0.0f) ? 0.0f : (1.0f / l_new);
      for (int d = 0; d < D; ++d) o[d] *= o_scale;
      for (int t = t0; t < t1; ++t) {
        if (sc[t - t0] == -INFINITY) continue;
        const float w = std::exp(sc[t - t0] - m_new) * b_scale;
        const float *v = V + (size_t)t * D;
        for (int d = 0; d < D; ++d) o[d] += w * v[d];
      }
      m = m_new;
      l = l_new;
    }
  }
}

}  // extern "C"
