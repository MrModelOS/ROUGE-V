/*
 * Темпоральный апскейлер — наш открытый аналог подхода, который лежит в
 * основе DLSS/FSR: реконструкция + накопление по времени + демоизинг.
 *
 * Важное различие, которое мы не прячем: у DLSS реконструкция нейросетевая
 * (веса обучены на данных NVIDIA). У нас — классическая: пространственная
 * реконструкция Catmull-Rom + репроекция истории по векторам движения +
 * клиппинг истории по окрестности текущего кадра + проверка глубины.
 * Это не «то же самое по качеству», зато это полностью наш код: никаких
 * чужих весов, никакой обученной сети, всё считается на CPU.
 *
 * Конвейер кадра:
 *   1) текущий кадр в низком разрешении -> 2x реконструкция (Catmull-Rom);
 *   2) векторы движения (в пикселях низкого разрешения) -> позиция в истории;
 *   3) история клиппится по AABB окрестности текущего кадра в YCoCg;
 *   4) если глубина в истории не совпадает с текущей — история отбрасывается
 *      (иначе «призраки» на краях движущихся объектов);
 *   5) смешивание: чем больше движение, тем меньше вес истории.
 *
 * Системы координат: индекс пикселя i занимает непрерывный отрезок
 * [i, i+1), центр — в i+0.5. Везде в файле используется именно она.
 */

#include "rouge/render.h"

#include <cmath>
#include <vector>

namespace {

inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Билинейная выборка в соглашении [i, i+1): пиксель i занимает [i, i+1),
// центр — в i+0.5. Поэтому индексы отсчётов считаются от floor(p - 0.5):
// при p = i+0.5 (центр пикселя i) получаем ровно пиксель i, без сдвига.
inline void sample_bilinear_rgb(const float *px, int w, int h, float fx, float fy,
                                 float *out) {
  const float gx = fx - 0.5f, gy = fy - 0.5f;
  const int x0 = clampi((int)std::floor(gx), 0, w - 1);
  const int y0 = clampi((int)std::floor(gy), 0, h - 1);
  const int x1 = clampi(x0 + 1, 0, w - 1);
  const int y1 = clampi(y0 + 1, 0, h - 1);
  const float tx = gx - std::floor(gx);
  const float ty = gy - std::floor(gy);
  const float *p00 = px + ((size_t)y0 * w + x0) * 3;
  const float *p10 = px + ((size_t)y0 * w + x1) * 3;
  const float *p01 = px + ((size_t)y1 * w + x0) * 3;
  const float *p11 = px + ((size_t)y1 * w + x1) * 3;
  for (int c = 0; c < 3; ++c) {
    const float a = p00[c] + (p10[c] - p00[c]) * tx;
    const float b = p01[c] + (p11[c] - p01[c]) * tx;
    out[c] = a + (b - a) * ty;
  }
}

inline float sample_bilinear_1(const float *px, int w, int h, float fx, float fy) {
  const float gx = fx - 0.5f, gy = fy - 0.5f;
  const int x0 = clampi((int)std::floor(gx), 0, w - 1);
  const int y0 = clampi((int)std::floor(gy), 0, h - 1);
  const int x1 = clampi(x0 + 1, 0, w - 1);
  const int y1 = clampi(y0 + 1, 0, h - 1);
  const float tx = gx - std::floor(gx);
  const float ty = gy - std::floor(gy);
  const float a = px[(size_t)y0 * w + x0] +
                  (px[(size_t)y0 * w + x1] - px[(size_t)y0 * w + x0]) * tx;
  const float b = px[(size_t)y1 * w + x0] +
                  (px[(size_t)y1 * w + x1] - px[(size_t)y1 * w + x0]) * tx;
  return a + (b - a) * ty;
}

// Catmull-Rom: 4x4 отсчёта, бикубическая реконструкция 2x.
// Веса Catmull-Rom для четырёх точек y0-1, y0, y0+1, y0+2 при доле t:
//   w(-1) = -0.5t(t-1)^2,  w(0) = 1 - t^2(2.5-1.5t),
//   w(1)  = 0.5t(1+t(4-3t)), w(2) = 0.5t^2(t-1).
// Сумма весов тождественно 1, при t=0 получаем точное попадание в отсчёт.
//
// jx/jy — смещение отсчёта от центра пикселя в низком разрешении (джиттер
// кадра). Реконструкция сдвигает свои точки выборки на это смещение, и
// картинка сразу попадает в «каноническую» сетку. Без этого история
// пришлось бы пересемплировать на полпикселя каждый кадр, и размытие
// накапливалось бы (это и есть «jitter-aware reconstruction» в TAAU).
inline void catmull_rom_2x(const float *src, int w, int h, float *dst,
                           float jx, float jy) {
  const int W = w * 2, H = h * 2;
  for (int Y = 0; Y < H; ++Y) {
    // Центр высокого пикселя Y в непрерывных координатах низкого разрешения.
    const float fy = (Y + 0.5f) * 0.5f - 0.5f - jy;
    const int y0 = (int)std::floor(fy);
    const float ty = fy - y0;
    const float wy[4] = {
        -0.5f * ty * (ty - 1.0f) * (ty - 1.0f),
        1.0f - ty * ty * (2.5f - 1.5f * ty),
        0.5f * ty * (1.0f + ty * (4.0f - 3.0f * ty)),
        0.5f * ty * ty * (ty - 1.0f)};
    for (int X = 0; X < W; ++X) {
      const float fx = (X + 0.5f) * 0.5f - 0.5f - jx;
      const int x0 = (int)std::floor(fx);
      const float tx = fx - x0;
      const float wx[4] = {
          -0.5f * tx * (tx - 1.0f) * (tx - 1.0f),
          1.0f - tx * tx * (2.5f - 1.5f * tx),
          0.5f * tx * (1.0f + tx * (4.0f - 3.0f * tx)),
          0.5f * tx * tx * (tx - 1.0f)};
      double acc[3] = {0.0, 0.0, 0.0};
      for (int j = 0; j < 4; ++j) {
        const int sy = clampi(y0 - 1 + j, 0, h - 1);
        const double wyj = wy[j];
        if (wyj == 0.0) continue;
        for (int i = 0; i < 4; ++i) {
          const int sx = clampi(x0 - 1 + i, 0, w - 1);
          const double wxy = wyj * wx[i];
          if (wxy == 0.0) continue;
          const float *p = src + ((size_t)sy * w + sx) * 3;
          acc[0] += wxy * p[0];
          acc[1] += wxy * p[1];
          acc[2] += wxy * p[2];
        }
      }
      float *o = dst + ((size_t)Y * W + X) * 3;
      o[0] = (float)acc[0];
      o[1] = (float)acc[1];
      o[2] = (float)acc[2];
    }
  }
}

// YCoCg (обратимое преобразование) — в нём удобно клиппить окрестность.
inline void rgb_to_ycocg(const float *rgb, float *ycc) {
  ycc[0] = 0.25f * rgb[0] + 0.5f * rgb[1] + 0.25f * rgb[2];
  ycc[1] = 0.5f * rgb[0] - 0.5f * rgb[2];
  ycc[2] = -0.25f * rgb[0] + 0.5f * rgb[1] - 0.25f * rgb[2];
}

inline void ycocg_to_rgb(const float *ycc, float *rgb) {
  rgb[0] = ycc[0] + ycc[1] - ycc[2];
  rgb[1] = ycc[0] + ycc[2];
  rgb[2] = ycc[0] - ycc[1] - ycc[2];
}

}  // namespace

extern "C" {

int rouge_upsample_bilinear(const RougeImage *low, RougeImage *out) {
  if (!low || !out) return -1;
  const int w = low->w, h = low->h;
  const int W = out->w, H = out->h;
  if (W < w || H < h) return -1;
  const float sx = (float)w / (float)W;
  const float sy = (float)h / (float)H;
  for (int Y = 0; Y < H; ++Y) {
    // Центр пикселя Y в соглашении [i, i+1) -> Y+0.5, в низком разрешении
    // это (Y+0.5)*sy.
    const float fy = ((float)Y + 0.5f) * sy;
    for (int X = 0; X < W; ++X) {
      const float fx = ((float)X + 0.5f) * sx;
      float c[3];
      sample_bilinear_rgb(low->px, w, h, fx, fy, c);
      float *o = out->px + ((size_t)Y * W + X) * 3;
      o[0] = c[0];
      o[1] = c[1];
      o[2] = c[2];
    }
  }
  return 0;
}

int rouge_upsample_temporal(const RougeImage *low, const RougeGBuffer *g,
                            const float *motion, const RougeImage *history,
                            const float *history_depth, float blend,
                            float clip_gamma, float jitter_x, float jitter_y,
                            RougeImage *out, RougeImage *new_history,
                            float *new_history_depth) {
  if (!low || !g || !motion || !out) return -1;
  const int w = low->w, h = low->h;
  const int W = out->w, H = out->h;
  if (W != w * 2 || H != h * 2) return -1;
  if (g->w != w || g->h != h) return -1;

  // 1) Пространственная реконструкция (с учётом джиттера кадра).
  std::vector<float> cur((size_t)W * H * 3);
  catmull_rom_2x(low->px, w, h, cur.data(), jitter_x, jitter_y);

  // Глубина текущего кадра в высоком разрешении.
  // Выборка в соглашении [i, i+1): центр высокого пикселя X -> низкий X/2+0.25.
  std::vector<float> depth((size_t)W * H);
  for (int Y = 0; Y < H; ++Y) {
    const float fy = (float)Y * 0.5f + 0.25f;
    for (int X = 0; X < W; ++X) {
      const float fx = (float)X * 0.5f + 0.25f;
      depth[(size_t)Y * W + X] = sample_bilinear_1(g->depth, w, h, fx, fy);
    }
  }

  for (int Y = 0; Y < H; ++Y) {
    for (int X = 0; X < W; ++X) {
      const size_t hi = (size_t)Y * W + X;
      float *dst = &cur[hi * 3];

      if (!history || !history->px || history->w != W || history->h != H) {
        // Первый кадр: истории нет — оставляем реконструкцию как есть.
        continue;
      }

      // Ближайший низкоразрешающий пиксель (для вектора движения).
      const int lx = clampi((int)std::floor((float)X * 0.5f + 0.5f), 0, w - 1);
      const int ly = clampi((int)std::floor((float)Y * 0.5f + 0.5f), 0, h - 1);
      const float mx = motion[((size_t)ly * w + lx) * 2 + 0];
      const float my = motion[((size_t)ly * w + lx) * 2 + 1];

      // 2) Репроекция: позиция в истории (в пикселях высокого разрешения).
      const float hx = (float)X + 0.5f - 2.0f * mx;
      const float hy = (float)Y + 0.5f - 2.0f * my;

      float hist[3];
      sample_bilinear_rgb(history->px, W, H, hx, hy, hist);

      // 4) Проверка глубины: история с другой поверхности не годится.
      if (history_depth) {
        const float hd = sample_bilinear_1(history_depth, W, H, hx, hy);
        const float cd = depth[hi];
        const float ref = 0.5f * (std::fabs(hd) + std::fabs(cd)) + 1e-4f;
        if (std::fabs(hd - cd) / ref > 0.1f) continue;  // история отброшена
      }

      // 3) Клиппинг по окрестности текущего кадра (3x3) в YCoCg.
      //    Важно: клиппим не по min/max, а по mean ± gamma*sigma (variance
      //    clipping). При min/max шумный кадр (1 spp) задавал бы слишком
      //    узкий коридор и history каждый кадр «съедался» бы — накопление
      //    переставало работать. Дисперсия по окрестности оценивает
      //    неопределённость и держит коридор адаптивным.
      float lo[3] = {1e30f, 1e30f, 1e30f};
      float hi_y[3] = {-1e30f, -1e30f, -1e30f};
      double mean[3] = {0.0, 0.0, 0.0};
      int cnt = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        const int ny = clampi(Y + dy, 0, H - 1);
        for (int dx = -1; dx <= 1; ++dx) {
          const int nx = clampi(X + dx, 0, W - 1);
          float ycc[3];
          rgb_to_ycocg(&cur[((size_t)ny * W + nx) * 3], ycc);
          for (int c = 0; c < 3; ++c) {
            if (ycc[c] < lo[c]) lo[c] = ycc[c];
            if (ycc[c] > hi_y[c]) hi_y[c] = ycc[c];
            mean[c] += ycc[c];
          }
          ++cnt;
        }
      }
      for (int c = 0; c < 3; ++c) mean[c] /= (double)cnt;
      double var[3] = {0.0, 0.0, 0.0};
      for (int dy = -1; dy <= 1; ++dy) {
        const int ny = clampi(Y + dy, 0, H - 1);
        for (int dx = -1; dx <= 1; ++dx) {
          const int nx = clampi(X + dx, 0, W - 1);
          float ycc[3];
          rgb_to_ycocg(&cur[((size_t)ny * W + nx) * 3], ycc);
          for (int c = 0; c < 3; ++c) {
            const double d = ycc[c] - mean[c];
            var[c] += d * d;
          }
        }
      }
      for (int c = 0; c < 3; ++c) var[c] /= (double)cnt;

      float hycc[3], hrgb[3];
      rgb_to_ycocg(hist, hycc);
      for (int c = 0; c < 3; ++c) {
        const double sigma = std::sqrt(var[c]);
        // Коридор: mean ± gamma*sigma (variance clipping). Шире min/max не
        // ограничиваем: при шумном кадре min/max слишком узок и history
        // каждый кадр срезался бы по краям, убивая накопление.
        const float clo = (float)(mean[c] - (double)clip_gamma * sigma);
        const float chi = (float)(mean[c] + (double)clip_gamma * sigma);
        if (hycc[c] < clo) hycc[c] = clo;
        if (hycc[c] > chi) hycc[c] = chi;
      }
      ycocg_to_rgb(hycc, hrgb);

      // 5) Смешивание: вес истории падает с ростом движения.
      const float motion_len = std::sqrt(mx * mx + my * my);
      float alpha = blend / (1.0f + motion_len);
      if (alpha < 0.0f) alpha = 0.0f;
      if (alpha > 1.0f) alpha = 1.0f;
      for (int c = 0; c < 3; ++c)
        dst[c] = dst[c] * (1.0f - alpha) + hrgb[c] * alpha;
    }
  }

  // Результат кадра — в out (и, если просят, в new_history).
  for (size_t i = 0; i < (size_t)W * H * 3; ++i) out->px[i] = cur[i];
  if (new_history && new_history->px && new_history->w == W &&
      new_history->h == H) {
    for (size_t i = 0; i < (size_t)W * H * 3; ++i) new_history->px[i] = cur[i];
  }
  if (new_history_depth) {
    for (size_t i = 0; i < (size_t)W * H; ++i) new_history_depth[i] = depth[i];
  }
  return 0;
}

}  // extern "C"
