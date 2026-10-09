/*
 * Свёртка Wave 2: conv2d NCHW через im2col + свой блочный GEMM.
 *
 * Прямая свёртка считается как матричное умножение: каждый входной образ
 * разворачивается в матрицу col [(Cin*KH*KW)][OH*OW]] (im2col), веса
 * перекладываются в [(Cout)][Cin*KH*KW], дальше обычный rouge_gemm_f32
 * (упаковка панелей, автотюнер — всё переиспользуется). Отдельного
 * микроядра под свёртку нет: это честный im2col+GEMM, не замаскированный
 * прямой алгоритм; производительность наследует GEMM.
 *
 * Соглашения: NCHW, row-major внутри каждой плоскости; веса
 * [Cout][Cin][KH][KW] подряд; stride/pad симметричные; bias может быть NULL.
 * Размеры на выходе: OH = (H + 2*pad - KH)/stride + 1 (целочисленно, остаток
 * отбрасывается — как в закрытых фреймворках; некорректные размеры дают
 * OH/OW <= 0 и пустой выход без записи).
 */

#include "rouge/numerics.h"

#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {

void rouge_conv2d_f32(int N, int Cin, int H, int W, int Cout, int KH, int KW,
                      int stride, int pad, const float *X, const float *Wt,
                      const float *bias, float *Y,
                      const RougeGemmConfig *cfg) {
  if (!X || !Wt || !Y) return;
  if (N <= 0 || Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || KH <= 0 ||
      KW <= 0 || stride <= 0 || pad < 0)
    return;
  const int OH = (H + 2 * pad - KH) / stride + 1;
  const int OW = (W + 2 * pad - KW) / stride + 1;
  if (OH <= 0 || OW <= 0) return;

  const size_t K = (size_t)Cin * (size_t)KH * (size_t)KW;  // глубина GEMM
  const size_t Ncol = (size_t)OH * (size_t)OW;             // ширина GEMM
  std::vector<float> col(K * Ncol);

  for (int n = 0; n < N; ++n) {
    // im2col одного образа: col[(c*KH*KW + kr*KW + kc)][oh*OW + ow].
    for (int c = 0; c < Cin; ++c)
      for (int kr = 0; kr < KH; ++kr)
        for (int kc = 0; kc < KW; ++kc) {
          const size_t row = ((size_t)c * KH + kr) * KW + kc;
          for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
              const int ih = oh * stride - pad + kr;
              const int iw = ow * stride - pad + kc;
              float v = 0.0f;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                v = X[((size_t)n * Cin + c) * H * W + (size_t)ih * W + iw];
              col[row * Ncol + (size_t)oh * OW + ow] = v;
            }
        }
    // Веса уже лежат как [Cout][K] подряд — GEMM напрямую, без переупаковки.
    float *Yn = Y + (size_t)n * Cout * OH * OW;
    rouge_gemm_f32(Cout, (int)Ncol, (int)K, Wt, col.data(), Yn, cfg);
    if (bias) {
      for (int co = 0; co < Cout; ++co)
        for (size_t i = 0; i < Ncol; ++i) Yn[(size_t)co * Ncol + i] += bias[co];
    }
  }
}

}  // extern "C"
