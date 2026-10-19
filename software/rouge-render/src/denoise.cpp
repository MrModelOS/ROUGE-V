/*
 * Денойзер: à-trous вейвлет-фильтр (B-spline 5 отсчётов) с направляющими
 * весами по нормали, относительной глубине и яркости.
 *
 * Это тот же по духу фильтр, что в SVGF-денойзере: разложение по
 * вейвлетам (шаг растёт с каждой итерацией) + веса, гасящие фильтрацию
 * через края и границы объектов. Отличие от SVGF: у нас нет отдельной
 * сети обучения весов — веса выведены из геометрии и дисперсии. Поэтому
 * всё считается на CPU, без обучающих данных и без чужого кода.
 *
 * Веса:
 *   w_n = max(0, n_p . n_q)^phi_normal   — не сглаживаем через углы;
 *   w_z = exp(-|z_p - z_q| / (phi_depth * z_сред + eps)) — относительная глубина;
 *   w_l = exp(-|l_p - l_q| / (phi_l * sqrt(var_p + var_q) + eps)) —
 *         яркость, отмасштабированная ошибкой оценки (иначе на гладких
 *         участках с большой дисперсией фильтр «съедал» бы детали).
 *
 * Дисперсия фильтруется тем же ядром: после каждой итерации оценка ошибки
 * становится уже, и яркостный вес начинает сильнее защищать края. Это
 * ровно тот приём, который применяется в SVGF для «пробуждения» весов на
 * деталях после первой итерации.
 */

#include "rouge/render.h"

#include <cmath>
#include <vector>

namespace {

// B-spline ядро à-trous: [1 4 6 4 1]/16.
const float kKernel[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f,
                          4.0f / 16.0f, 1.0f / 16.0f};

inline float luminance(const float *px) {
  return 0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];
}

inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace

extern "C" {

int rouge_denoise_atrous(const RougeImage *color, const RougeGBuffer *g,
                         const float *variance, int iterations, float phi_normal,
                         float phi_depth, float phi_variance, RougeImage *out) {
  if (!color || !g || !out) return -1;
  if (color->w != out->w || color->h != out->h) return -1;
  if (g->w != color->w || g->h != color->h) return -1;
  const int w = color->w, h = color->h;
  const size_t n = (size_t)w * h;

  std::vector<float> cur((size_t)w * h * 3);
  for (size_t i = 0; i < cur.size(); ++i) cur[i] = color->px[i];

  if (iterations <= 0) {
    for (size_t i = 0; i < cur.size(); ++i) out->px[i] = cur[i];
    return 0;
  }

  // Текущая оценка дисперсии (на канал; для весов берём канал R — он
  // один и тот же для всех, чтобы веса не «мигали» между каналами).
  std::vector<float> var(n);
  for (size_t i = 0; i < n; ++i)
    var[i] = variance ? variance[i * 3] : 0.0f;

  std::vector<float> next((size_t)w * h * 3);
  std::vector<float> var_next(n);
  std::vector<float> lum(n);

  for (int iter = 0; iter < iterations; ++iter) {
    const int step = 1 << iter;  // 1, 2, 4, 8...
    for (size_t i = 0; i < n; ++i) lum[i] = luminance(&cur[i * 3]);

    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const size_t pi = (size_t)y * w + x;
        const float *n_p = &g->normal[pi * 3];
        const float z_p = g->depth[pi];
        const float l_p = lum[pi];
        const float var_p = var[pi];

        double wsum = 0.0;
        double csum[3] = {0.0, 0.0, 0.0};
        double vsum = 0.0;
        for (int dy = -2; dy <= 2; ++dy) {
          const int qy = clampi(y + dy * step, 0, h - 1);
          for (int dx = -2; dx <= 2; ++dx) {
            const int qx = clampi(x + dx * step, 0, w - 1);
            const size_t qi = (size_t)qy * w + qx;
            const float kw = kKernel[dx + 2] * kKernel[dy + 2];
            if (kw <= 0.0f) continue;

            const float *n_q = &g->normal[qi * 3];
            const float z_q = g->depth[qi];
            const float l_q = lum[qi];
            const float var_q = var[qi];

            // Нормаль: гасим через угол между нормалями.
            const float ndot =
                n_p[0] * n_q[0] + n_p[1] * n_q[1] + n_p[2] * n_q[2];
            float w = 1.0f;
            if (phi_normal > 0.0f)
              w = std::pow(ndot > 0.0f ? ndot : 0.0f, phi_normal);

            // Относительная глубина: |z_p - z_q| / средняя глубина.
            const float zrel = std::fabs(z_p - z_q);
            const float zref =
                0.5f * (std::fabs(z_p) + std::fabs(z_q)) + 1e-4f;
            w *= std::exp(-zrel / (phi_depth * zref));

            // Яркость, отмасштабированная ошибкой оценки.
            const float err = std::sqrt(std::fmax(0.0f, var_p + var_q));
            w *= std::exp(-std::fabs(l_p - l_q) / (phi_variance * err + 1e-4f));

            const float ww = kw * w;
            wsum += ww;
            for (int c = 0; c < 3; ++c)
              csum[c] += ww * (double)cur[qi * 3 + c];
            vsum += ww * (double)var_q;
          }
        }
        if (wsum > 0.0) {
          for (int c = 0; c < 3; ++c)
            next[pi * 3 + c] = (float)(csum[c] / wsum);
          var_next[pi] = (float)(vsum / wsum);
        } else {
          for (int c = 0; c < 3; ++c) next[pi * 3 + c] = cur[pi * 3 + c];
          var_next[pi] = var_p;
        }
      }
    }
    cur.swap(next);
    var.swap(var_next);
  }

  for (size_t i = 0; i < cur.size(); ++i) out->px[i] = cur[i];
  return 0;
}

}  // extern "C"
