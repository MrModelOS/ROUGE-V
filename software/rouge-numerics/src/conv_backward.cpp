/*
 * Backward свёртки Wave 2 (для обучения): градиенты прямыми циклами.
 *
 * Forward: Y[n,co,oh,ow] = sum_{ci,kr,kc} X[n,ci,oh*s-p+kr,ow*s-p+kc] * W.
 *   dx[n,ci,ih,iw] = sum_{co,kr,kc} sum_{(oh,ow): oh*s-p+kr=ih, ...}
 *                     dy[n,co,oh,ow] * W[co,ci,kr,kc]   (транспонированная)
 *   dw[co,ci,kr,kc] = sum_{n,oh,ow} dy[n,co,oh,ow] *
 *                     X[n,ci,oh*s-p+kr,ow*s-p+kc]        (корреляция)
 * Сложность та же, что forward; никакой отдельной математики — цепное правило
 * по определению forward. Тест — против численного градиента (центральные
 * разности), т.е. независимого определения производной.
 * Некорректные размеры или NULL -> ничего не делать.
 */

#include "rouge/numerics.h"

#include <cstddef>

extern "C" {

void rouge_conv_bwd_data(int N, int Cin, int H, int W, int Cout, int KH, int KW,
                         int stride, int pad, const float *dy, const float *Wt,
                         float *dx) {
  if (!dy || !Wt || !dx) return;
  if (N <= 0 || Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || KH <= 0 ||
      KW <= 0 || stride <= 0 || pad < 0)
    return;
  const int OH = (H + 2 * pad - KH) / stride + 1;
  const int OW = (W + 2 * pad - KW) / stride + 1;
  if (OH <= 0 || OW <= 0) return;
  for (int n = 0; n < N; ++n)
    for (int ci = 0; ci < Cin; ++ci)
      for (int ih = 0; ih < H; ++ih)
        for (int iw = 0; iw < W; ++iw) {
          double s = 0.0;
          for (int co = 0; co < Cout; ++co)
            for (int kr = 0; kr < KH; ++kr) {
              const int num_h = ih + pad - kr;
              if (num_h % stride != 0) continue;
              const int oh = num_h / stride;
              if (oh < 0 || oh >= OH) continue;
              for (int kc = 0; kc < KW; ++kc) {
                const int num_w = iw + pad - kc;
                if (num_w % stride != 0) continue;
                const int ow = num_w / stride;
                if (ow < 0 || ow >= OW) continue;
                s += (double)dy[((size_t)n * Cout + co) * OH * OW + oh * OW + ow] *
                     (double)Wt[((size_t)co * Cin + ci) * KH * KW + kr * KW + kc];
              }
            }
          dx[((size_t)n * Cin + ci) * H * W + (size_t)ih * W + iw] = (float)s;
        }
}

void rouge_conv_bwd_filter(int N, int Cin, int H, int W, int Cout, int KH,
                           int KW, int stride, int pad, const float *X,
                           const float *dy, float *dW) {
  if (!X || !dy || !dW) return;
  if (N <= 0 || Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || KH <= 0 ||
      KW <= 0 || stride <= 0 || pad < 0)
    return;
  const int OH = (H + 2 * pad - KH) / stride + 1;
  const int OW = (W + 2 * pad - KW) / stride + 1;
  if (OH <= 0 || OW <= 0) return;
  for (int co = 0; co < Cout; ++co)
    for (int ci = 0; ci < Cin; ++ci)
      for (int kr = 0; kr < KH; ++kr)
        for (int kc = 0; kc < KW; ++kc) {
          double s = 0.0;
          for (int n = 0; n < N; ++n)
            for (int oh = 0; oh < OH; ++oh)
              for (int ow = 0; ow < OW; ++ow) {
                const int ih = oh * stride - pad + kr;
                const int iw = ow * stride - pad + kc;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                s += (double)dy[((size_t)n * Cout + co) * OH * OW + oh * OW + ow] *
                     (double)X[((size_t)n * Cin + ci) * H * W + ih * W + iw];
              }
          dW[((size_t)co * Cin + ci) * KH * KW + (size_t)kr * KW + kc] = (float)s;
        }
}

}  // extern "C"
