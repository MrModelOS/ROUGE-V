/*
 * Winograd Wave 2: F(2x2,3x3) для свёртки 3x3 stride 1 pad 1.
 *
 * Матрицы — классические Lavin & Gray F(2,3) (общеизвестные константы):
 *   B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
 *   G   = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]]
 *   A^T = [[1,1,1,0],[0,1,-1,-1]]
 * Плитка выхода 2x2 из входа 4x4: Y = A^T ( (G g G^T) ⊙ (B^T d B) ) A.
 * Суммирование по Cin — до A-преобразования (в пространстве Winograd).
 * Все 1/2 точны в f32; арифметика обычная float.
 *
 * Границы: работает при stride==1 && pad==1 && KH==KW==3, иначе возврат -1
 * (вызывающий падает на im2col — честный fallback, не тихий мискомпил).
 * Нечётные OH/OW: полные 2x2-плитки — Winograd, последний ряд/столбец —
 * прямой досчёт тем же вызовом. N — любой, bias NULL-able.
 */

#include "rouge/numerics.h"

#include <chrono>
#include <cstring>
#include <vector>

namespace {

// G*g*G^T для фильтра 3x3 -> 4x4. g row-major 3x3.
void filter_xform(const float *g, float U[4][4]) {
  // Gw = G*g (4x3), затем U = Gw*G^T (4x4).
  float Gw[4][3];
  for (int j = 0; j < 3; ++j) {
    Gw[0][j] = g[0 * 3 + j];
    Gw[1][j] = 0.5f * (g[0 * 3 + j] + g[1 * 3 + j] + g[2 * 3 + j]);
    Gw[2][j] = 0.5f * (g[0 * 3 + j] - g[1 * 3 + j] + g[2 * 3 + j]);
    Gw[3][j] = g[2 * 3 + j];
  }
  for (int i = 0; i < 4; ++i) {
    U[i][0] = Gw[i][0];
    U[i][1] = 0.5f * (Gw[i][0] + Gw[i][1] + Gw[i][2]);
    U[i][2] = 0.5f * (Gw[i][0] - Gw[i][1] + Gw[i][2]);
    U[i][3] = Gw[i][2];
  }
}

// V = B^T*d*B для входа 4x4 -> 4x4.
void input_xform(const float d[4][4], float V[4][4]) {
  float T[4][4];  // T = B^T * d
  for (int j = 0; j < 4; ++j) {
    T[0][j] = d[0][j] - d[2][j];
    T[1][j] = d[1][j] + d[2][j];
    T[2][j] = -d[1][j] + d[2][j];
    T[3][j] = d[1][j] - d[3][j];
  }
  // V = T * B
  for (int i = 0; i < 4; ++i) {
    V[i][0] = T[i][0] - T[i][2];
    V[i][1] = T[i][1] + T[i][2];
    V[i][2] = -T[i][1] + T[i][2];
    V[i][3] = T[i][1] - T[i][3];
  }
}

// Y(2x2) = A^T*M*A.
void output_xform(const float M[4][4], float Y[2][2]) {
  float T[2][4];  // T = A^T * M
  for (int j = 0; j < 4; ++j) {
    T[0][j] = M[0][j] + M[1][j] + M[2][j];
    T[1][j] = M[1][j] - M[2][j] - M[3][j];
  }
  for (int i = 0; i < 2; ++i) {
    Y[i][0] = T[i][0] + T[i][1] + T[i][2];
    Y[i][1] = T[i][1] - T[i][2] - T[i][3];
  }
}

}  // namespace

extern "C" {

int rouge_conv_winograd3x3_f32(int N, int Cin, int H, int W, int Cout,
                               const float *X, const float *Wt,
                               const float *bias, float *Y) {
  if (!X || !Wt || !Y) return -1;
  if (N <= 0 || Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0) return -1;
  const int OH = H, OW = W;  // 3x3 s1 p1 сохраняет размер
  // U заранее: [Cout][Cin][4][4].
  std::vector<float> U((size_t)Cout * Cin * 16);
  for (int co = 0; co < Cout; ++co)
    for (int ci = 0; ci < Cin; ++ci) {
      float M4[4][4];
      filter_xform(Wt + ((size_t)co * Cin + ci) * 9, M4);
      std::memcpy(&U[((size_t)co * Cin + ci) * 16], M4, sizeof(M4));
    }

  const int TH = OH / 2, TW = OW / 2;  // полные плитки
  // V-трансформы входных плиток: один раз на (n, плитка, ci), переиспользуются
  // всеми Cout. Первая версия считала их внутри цикла по co — Winograd
  // проигрывал im2col в Cout раз (замер: 3.72 мс против 0.47 мс).
  std::vector<float> Vbuf((size_t)Cin * 16);
  for (int n = 0; n < N; ++n) {
    for (int th = 0; th < TH; ++th)
      for (int tw = 0; tw < TW; ++tw) {
        for (int ci = 0; ci < Cin; ++ci) {
          // Входная плитка 4x4 с паддингом 1.
          float dd[4][4];
          for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
              const int ih = th * 2 + i - 1, iw = tw * 2 + j - 1;
              float v = 0.0f;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                v = X[((size_t)n * Cin + ci) * H * W + ih * W + iw];
              dd[i][j] = v;
            }
          float VV[4][4];
          input_xform(dd, VV);
          std::memcpy(&Vbuf[(size_t)ci * 16], VV, sizeof(VV));
        }
        for (int co = 0; co < Cout; ++co) {
          float M[4][4] = {};
          for (int ci = 0; ci < Cin; ++ci) {
            const float *UU = &U[((size_t)co * Cin + ci) * 16];
            const float *VV = &Vbuf[(size_t)ci * 16];
            for (int k = 0; k < 16; ++k) M[k / 4][k % 4] += UU[k] * VV[k];
          }
          float Yt[2][2];
          output_xform(M, Yt);
          float *Yn = Y + ((size_t)n * Cout + co) * (size_t)OH * OW;
          const float b = bias ? bias[co] : 0.0f;
          Yn[(th * 2 + 0) * OW + tw * 2 + 0] = Yt[0][0] + b;
          Yn[(th * 2 + 0) * OW + tw * 2 + 1] = Yt[0][1] + b;
          Yn[(th * 2 + 1) * OW + tw * 2 + 0] = Yt[1][0] + b;
          Yn[(th * 2 + 1) * OW + tw * 2 + 1] = Yt[1][1] + b;
        }
      }
    // Хвосты при нечётных OH/OW — прямой досчёт.
    for (int co = 0; co < Cout; ++co) {
      float *Yn = Y + ((size_t)n * Cout + co) * (size_t)OH * OW;
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
          if (oh < TH * 2 && ow < TW * 2) continue;
          double s = bias ? bias[co] : 0.0;
          for (int ci = 0; ci < Cin; ++ci)
            for (int kr = 0; kr < 3; ++kr)
              for (int kc = 0; kc < 3; ++kc) {
                const int ih = oh - 1 + kr, iw = ow - 1 + kc;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                s += (double)X[((size_t)n * Cin + ci) * H * W + ih * W + iw] *
                     (double)Wt[((size_t)co * Cin + ci) * 9 + kr * 3 + kc];
              }
          Yn[oh * OW + ow] = (float)s;
        }
    }
  }
  return 0;
}

// Автотюнер свёртки: im2col против Winograd (когда применим). Замер на
// настоящих данных, кэша нет, в Y — результат победителя. Возвращает профиль:
// method 0=im2col, 1=winograd, иначе -1 при некорректных размерах.
RougeConvProfile rouge_conv_tuned(int N, int Cin, int H, int W, int Cout,
                                  int KH, int KW, int stride, int pad,
                                  const float *X, const float *Wt,
                                  const float *bias, float *Y) {
  RougeConvProfile p;
  p.method = -1;
  p.millis = 0.0;
  if (!X || !Wt || !Y) return p;
  if (N <= 0 || Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || KH <= 0 ||
      KW <= 0 || stride <= 0 || pad < 0)
    return p;
  const int OH = (H + 2 * pad - KH) / stride + 1;
  const int OW = (W + 2 * pad - KW) / stride + 1;
  if (OH <= 0 || OW <= 0) return p;
  const size_t out_n = (size_t)N * Cout * OH * OW;
  std::vector<float> alt(out_n);
  const bool can_wino =
      (KH == 3 && KW == 3 && stride == 1 && pad == 1 && N >= 1);

  auto timed = [&](int method) {
    auto t0 = std::chrono::steady_clock::now();
    if (method == 1)
      rouge_conv_winograd3x3_f32(N, Cin, H, W, Cout, X, Wt, bias,
                                 method == 1 ? alt.data() : Y);
    else
      rouge_conv2d_f32(N, Cin, H, W, Cout, KH, KW, stride, pad, X, Wt, bias,
                       method == 0 ? Y : alt.data(), nullptr);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };
  // Разогрев вне замера.
  rouge_conv2d_f32(N, Cin, H, W, Cout, KH, KW, stride, pad, X, Wt, bias, Y,
                   nullptr);
  double t_im2col = timed(0);
  // Второй прогон im2col — минимум из двух (шум планировщика).
  double t2 = timed(0);
  if (t2 < t_im2col) t_im2col = t2;
  p.method = 0;
  p.millis = t_im2col;
  if (can_wino) {
    rouge_conv_winograd3x3_f32(N, Cin, H, W, Cout, X, Wt, bias, alt.data());
    double t_w = timed(1);
    double t_w2 = timed(1);
    if (t_w2 < t_w) t_w = t_w2;
    if (t_w < t_im2col) {
      std::memcpy(Y, alt.data(), out_n * sizeof(float));
      p.method = 1;
      p.millis = t_w;
    }
  }
  return p;
}

}  // extern "C"
