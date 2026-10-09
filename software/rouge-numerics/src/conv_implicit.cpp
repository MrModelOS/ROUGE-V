/*
 * Implicit-GEMM свёртка Wave 2: прямой тайловый проход без col-буфера.
 *
 * В отличие от im2col (conv.cpp), входная матрица col[K][Ncol] никогда не
 * материализуется: тайл выхода [tc][ts] накапливается прямо из X через
 * индексную арифметику (ih = oh*stride-pad+kr), веса читаются тайлами
 * [tc] на каждую позицию (ci,kr,kc). Сложность та же O(Cout*OH*OW*Cin*KH*KW),
 * память — O(tc*ts) вместо O(K*Ncol). tile_m/tile_n из cfg задают tc/ts
 * (clamp 1..64, умолчания 8/128); остальные поля cfg игнорируются.
 *
 * Это честный implicit: тот же счёт, что im2col+GEMM, другим порядком
 * обхода. Точность — тот же float-порядок накопления, расхождение с im2col
 * только от перестановки слагаемых (тест: допуск 1e-5).
 */

#include "rouge/numerics.h"

#include <algorithm>
#include <cstddef>
#include <vector>

extern "C" {

void rouge_conv_implicit_f32(int N, int Cin, int H, int W, int Cout, int KH,
                             int KW, int stride, int pad, const float *X,
                             const float *Wt, const float *bias, float *Y,
                             const RougeGemmConfig *cfg) {
  if (!X || !Wt || !Y) return;
  if (N <= 0 || Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || KH <= 0 ||
      KW <= 0 || stride <= 0 || pad < 0)
    return;
  const int OH = (H + 2 * pad - KH) / stride + 1;
  const int OW = (W + 2 * pad - KW) / stride + 1;
  if (OH <= 0 || OW <= 0) return;

  int tc = 8, ts = 128;
  if (cfg) {
    if (cfg->tile_m > 0) tc = std::min(cfg->tile_m, 64);
    if (cfg->tile_n > 0) ts = std::min(cfg->tile_n, 256);
    if (tc < 1) tc = 1;
    if (ts < 1) ts = 1;
  }
  const int Ncol = OH * OW;
  std::vector<float> acc((size_t)tc * ts);
  std::vector<float> wbuf(tc);

  for (int n = 0; n < N; ++n) {
    for (int co0 = 0; co0 < Cout; co0 += tc) {
      const int ctc = std::min(tc, Cout - co0);
      for (int s0 = 0; s0 < Ncol; s0 += ts) {
        const int cts = std::min(ts, Ncol - s0);
        std::fill(acc.begin(), acc.begin() + (size_t)ctc * cts, 0.0f);
        for (int ci = 0; ci < Cin; ++ci)
          for (int kr = 0; kr < KH; ++kr)
            for (int kc = 0; kc < KW; ++kc) {
              for (int t = 0; t < ctc; ++t)
                wbuf[t] =
                    Wt[((size_t)(co0 + t) * Cin + ci) * KH * KW + kr * KW + kc];
              for (int s = 0; s < cts; ++s) {
                const int oh = (s0 + s) / OW, ow = (s0 + s) % OW;
                const int ih = oh * stride - pad + kr;
                const int iw = ow * stride - pad + kc;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                const float x =
                    X[((size_t)n * Cin + ci) * H * W + (size_t)ih * W + iw];
                for (int t = 0; t < ctc; ++t)
                  acc[(size_t)t * cts + s] += wbuf[t] * x;
              }
            }
        float *Yb = Y + ((size_t)n * Cout + co0) * Ncol + s0;
        for (int t = 0; t < ctc; ++t) {
          const float b = bias ? bias[co0 + t] : 0.0f;
          for (int s = 0; s < cts; ++s)
            Yb[(size_t)t * Ncol + s] = acc[(size_t)t * cts + s] + b;
        }
      }
    }
  }
}

}  // extern "C"
