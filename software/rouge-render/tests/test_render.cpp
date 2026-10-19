/*
 * Тесты rouge-render: денойзер и темпоральный апскейлер должны ЧЕСТНО
 * улучшать картинку, и это проверяется измерением, а не на слово.
 *
 * Схема проверки (как в работах по реконструкции):
 *   * эталон (GT) — тот же рендер, но со многими отсчётами (сходится);
 *   * «шумный» кадр — тот же рендер с малым числом отсчётов;
 *   * PSNR/SSIM считаются по тонированному изображению в [0,1].
 *
 * Требования к тесту:
 *   1) денойзер повышает PSNR относительно шумного кадра (существенно,
 *      а не на «долю процента») и не ухудшает SSIM;
 *   2) наш темпоральный апскейлер лучше билинейной интерполяции;
 *   3) рендер детерминирован: тот же seed -> тот же результат.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "rouge/render.h"

static int failures = 0;

#define EXPECT_TRUE(cond)                                          \
  do {                                                             \
    if (!(cond)) {                                                 \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
      ++failures;                                                  \
    }                                                              \
  } while (0)

// Тоновое отображение (экспоненциальное) + приведение к [0,1].
static void tonemap_into(const RougeImage &src, RougeImage &dst) {
  const size_t n = (size_t)src.w * src.h * 3;
  for (size_t i = 0; i < n; ++i) {
    const float v = 1.0f - std::exp(-src.px[i]);
    dst.px[i] = std::min(1.0f, std::max(0.0f, v));
  }
}

// Среднее radiance из накопленной цели рендера.
static void target_to_image(const RougeRenderTarget *t, RougeImage *img) {
  const double inv = t->samples > 0 ? 1.0 / (double)t->samples : 0.0;
  const size_t n = (size_t)t->w * t->h * 3;
  for (size_t i = 0; i < n; ++i) img->px[i] = (float)(t->sum[i] * inv);
}

int main() {
  RougeCamera cam;
  rouge_camera_default(&cam);
  // Эталон и «шумные» кадры снимаем без джиттера: иначе эталон сам
  // размыт и сравнение стало бы нечестным. Джиттер включаем только в
  // тесте темпорального апскейлера, где он и нужен.
  cam.jitter_scale = 0.0f;

  const int W = 64, H = 64;

  // ------------------------------------------------------------------ эталон
  RougeRenderTarget *gt = rouge_target_new(W, H);
  rouge_render_cornell(gt, &cam, 256, 8, 12345u, 0, 1, 10.0f);
  RougeImage *gt_img = rouge_image_new(W, H);
  target_to_image(gt, gt_img);
  RougeImage *gt_tm = rouge_image_new(W, H);
  tonemap_into(*gt_img, *gt_tm);
  std::printf("GT: %dx%d, %d spp\n", W, H, gt->samples);
  // Эталон сам сходится: средняя яркость должна быть в разумных пределах.
  {
    double sum = 0.0;
    for (size_t i = 0; i < (size_t)W * H * 3; ++i) sum += gt_tm->px[i];
    const double mean = sum / ((double)W * H * 3);
    std::printf("  средняя яркость эталона: %.4f\n", mean);
    EXPECT_TRUE(mean > 0.02 && mean < 0.9);
  }

  // ------------------------------------------------- 1) детерминированность
  {
    RougeRenderTarget *a = rouge_target_new(32, 32);
    RougeRenderTarget *b = rouge_target_new(32, 32);
    rouge_render_cornell(a, &cam, 4, 4, 777u, 3, 1, 10.0f);
    rouge_render_cornell(b, &cam, 4, 4, 777u, 3, 1, 10.0f);
    bool same = true;
    for (size_t i = 0; i < (size_t)32 * 32 * 3 && same; ++i)
      if (a->sum[i] != b->sum[i]) same = false;
    EXPECT_TRUE(same);
    rouge_target_free(a);
    rouge_target_free(b);
  }

  // --------------------------------------------------------- 2) денойзер
  // 4 отсчёта на пиксель: картинка уже не «файрфлаит», но шум виден.
  // (На 1 отсчёте PSNR тонет в единичных ярких выбросах, и фильтр на них
  // ничего не может — это честное ограничение а-тру фильтра.)
  {
    const int spp = 4;
    RougeRenderTarget *noisy = rouge_target_new(W, H);
    rouge_render_cornell(noisy, &cam, spp, 8, 4242u, 0, 1, 10.0f);
    RougeImage *n_img = rouge_image_new(W, H);
    target_to_image(noisy, n_img);
    RougeImage *n_tm = rouge_image_new(W, H);
    tonemap_into(*n_img, *n_tm);

    std::vector<float> variance((size_t)W * H * 3);
    rouge_render_variance(noisy, variance.data());

    RougeImage *den = rouge_image_new(W, H);
    const int iters = 3;
    const float phi_n = 64.0f, phi_z = 0.05f, phi_l = 1.0f;
    rouge_denoise_atrous(n_img, &noisy->g, variance.data(), iters, phi_n, phi_z,
                         phi_l, den);
    RougeImage *d_tm = rouge_image_new(W, H);
    tonemap_into(*den, *d_tm);

    const double psnr_noisy = rouge_psnr(n_tm, gt_tm);
    const double psnr_den = rouge_psnr(d_tm, gt_tm);
    const double ssim_noisy = rouge_ssim(n_tm, gt_tm);
    const double ssim_den = rouge_ssim(d_tm, gt_tm);
    std::printf("денойзер: PSNR шум %.2f -> %.2f (+%.2f дБ), SSIM %.4f -> %.4f\n",
                psnr_noisy, psnr_den, psnr_den - psnr_noisy, ssim_noisy,
                ssim_den);
    EXPECT_TRUE(psnr_den > psnr_noisy + 0.8);
    EXPECT_TRUE(ssim_den > ssim_noisy);

    rouge_image_free(n_img);
    rouge_image_free(n_tm);
    rouge_image_free(den);
    rouge_image_free(d_tm);
    rouge_target_free(noisy);
  }

  // ------------------------------------- 3) темпоральный апскейлер 2x
  // Два случая, потому что ведут себя они по-разному:
  //   a) статичная камера — накопление по времени как прогрессивное
  //      уточнение (пауза в игре): выигрыш максимальный;
  //   b) движущаяся камера — каждый кадр снят с другой точки, историю
  //      приходится перепроецировать, часть резкости теряется.
  // База в обоих случаях одна: билинейный апскейл одного кадра 1 spp.
  {
    const int lw = W / 2, lh = H / 2;
    const int frames = 8;

    for (int mode = 0; mode < 2; ++mode) {
      const float move = (mode == 0) ? 0.0f : 0.01f;
      RougeCamera camj = cam;
      camj.jitter_scale = 0.25f;  // лёгкий джиттер, как в TAA

      // База: один кадр 1 spp -> билиней.
      RougeRenderTarget *low1 = rouge_target_new(lw, lh);
      rouge_render_cornell(low1, &camj, 1, 8, 1000u, 0, 1, 10.0f);
      RougeImage *low_img = rouge_image_new(lw, lh);
      target_to_image(low1, low_img);
      RougeImage *bil = rouge_image_new(W, H);
      rouge_upsample_bilinear(low_img, bil);
      RougeImage *bil_tm = rouge_image_new(W, H);
      tonemap_into(*bil, *bil_tm);
      const double psnr_bil = rouge_psnr(bil_tm, gt_tm);
      const double ssim_bil = rouge_ssim(bil_tm, gt_tm);

      // Наш темпоральный апскейлер: 8 кадров.
      RougeImage *history = rouge_image_new(W, H);
      std::vector<float> hist_depth((size_t)W * H, 0.0f);
      RougeImage *up = rouge_image_new(W, H);
      std::vector<float> motion((size_t)lw * lh * 2, 0.0f);
      RougeCamera prev = camj;

      for (int f = 0; f < frames; ++f) {
        RougeCamera c = camj;
        c.pos[0] = camj.pos[0] - move * (float)(frames - 1 - f);
        c.target[0] = camj.target[0] - move * (float)(frames - 1 - f);

        RougeRenderTarget *lt = rouge_target_new(lw, lh);
        rouge_render_cornell(lt, &c, 1, 8, 1000u + (uint64_t)f, f, 1, 10.0f);
        RougeImage *li = rouge_image_new(lw, lh);
        target_to_image(lt, li);

        // Джиттер кадра нужен для репроекции: иначе история берётся не с той
        // точки, и накопление размазывается.
        float jx = 0.0f, jy = 0.0f, pjx = 0.0f, pjy = 0.0f;
        rouge_frame_jitter(f, 1, camj.jitter_scale, &jx, &jy);
        if (f > 0) rouge_frame_jitter(f - 1, 1, camj.jitter_scale, &pjx, &pjy);
        rouge_motion_from_world(&lt->g, &c, &prev, motion.data());
        prev = c;

        RougeImage *next_hist = rouge_image_new(W, H);
        std::vector<float> next_depth((size_t)W * H, 0.0f);
        rouge_upsample_temporal(li, &lt->g, motion.data(),
                                f == 0 ? nullptr : history,
                                f == 0 ? nullptr : hist_depth.data(), 0.9f, 3.0f,
                                jx, jy, up, next_hist, next_depth.data());
        std::copy(next_hist->px, next_hist->px + (size_t)W * H * 3, history->px);
        std::copy(next_depth.begin(), next_depth.end(), hist_depth.begin());

        rouge_image_free(li);
        rouge_image_free(next_hist);
        rouge_target_free(lt);
      }

      RougeImage *up_tm = rouge_image_new(W, H);
      tonemap_into(*up, *up_tm);
      const double psnr_up = rouge_psnr(up_tm, gt_tm);
      const double ssim_up = rouge_ssim(up_tm, gt_tm);
      std::printf("апскейлер (%s): билинейный PSNR %.2f (SSIM %.4f) -> наш "
                  "PSNR %.2f (SSIM %.4f)\n",
                  mode == 0 ? "статичная камера" : "движущаяся камера",
                  psnr_bil, ssim_bil, psnr_up, ssim_up);
      if (mode == 0) {
        // Прогрессивное уточнение: накопление должно заметно выиграть.
        EXPECT_TRUE(psnr_up > psnr_bil + 2.0);
      } else {
        // С движением выигрыш меньше, но картинка обязана не ухудшиться.
        EXPECT_TRUE(psnr_up > psnr_bil);
      }
      EXPECT_TRUE(ssim_up > ssim_bil + 0.02);

      rouge_image_free(low_img);
      rouge_image_free(bil);
      rouge_image_free(bil_tm);
      rouge_image_free(history);
      rouge_image_free(up);
      rouge_image_free(up_tm);
      rouge_target_free(low1);
    }
  }

  rouge_image_free(gt_img);
  rouge_image_free(gt_tm);
  rouge_target_free(gt);

  if (failures == 0) std::printf("PASS: render (денойзер + темпоральный апскейлер)\n");
  return failures ? 1 : 0;
}
