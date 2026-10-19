/*
 * rouge-render — своя графика: path tracer, денойзер и темпоральный
 * апскейлер (Wave 4: «свой вместо OptiX/DLSS»).
 *
 * Идея: DLSS — это не «магия», а связка из трёх вещей: реконструкция
 * (нейросеть у NVIDIA, у нас — классическая), накопление по времени и
 * демоизинг. Ни одну из чужих библиотек мы не зовём: свой трассировщик,
 * свой денойзер, свой апскейлер. Поэтому результат можно измерить честно:
 * есть эталон (тот же рендер со многими отсчётами), есть метрики (PSNR,
 * SSIM), и видно, что наш фильтр действительно лучше шума/билинейности.
 *
 * Никаких зависимостей от NVIDIA/OptiX/DLSS: ни кода, ни бинарников, ни
 * форматов. Только свой код на C++17.
 *
 * Все структуры — POD, API в стиле extern "C" (как в остальном стеке).
 */

#ifndef ROUGE_RENDER_H
#define ROUGE_RENDER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Изображение: RGB float32, row-major, w*h*3 --- */
typedef struct {
  int w, h;
  float *px;
} RougeImage;

/* --- G-буфер: то, что нужно денойзеру и репроекции --- */
typedef struct {
  int w, h;
  float *normal;   /* w*h*3 — нормаль первой поверхности */
  float *depth;    /* w*h   — расстояние до первой поверхности */
  float *albedo;   /* w*h*3 — альбедо первой поверхности */
  float *position; /* w*h*3 — мировая точка первого пересечения */
} RougeGBuffer;

/* --- Камера --- */
typedef struct {
  float pos[3];
  float target[3];
  float up[3];
  float fov_y;        /* вертикальный угол обзора, градусы */
  float jitter_scale; /* амплитуда джиттера в пикселях (0 = без джиттера) */
} RougeCamera;

/* --- Цель рендера: накопление отсчётов + G-буфер --- */
typedef struct {
  int w, h;
  float *sum;   /* w*h*3 — сумма radiance по отсчётам */
  float *sumsq; /* w*h*3 — сумма квадратов (для дисперсии) */
  int samples;  /* сколько отсчётов учтено */
  RougeGBuffer g;
} RougeRenderTarget;

/* --- Время (жизненный цикл) --- */

RougeImage *rouge_image_new(int w, int h);
void rouge_image_free(RougeImage *img);
RougeRenderTarget *rouge_target_new(int w, int h);
void rouge_target_free(RougeRenderTarget *t);
void rouge_camera_default(RougeCamera *cam);

/* --- Рендер ---

 * rouge_render_cornell: трассировка Cornell-box (диффузная сцена с
 * площадным источником) с next-event estimation и MIS (power heuristic),
 * косинусным сэмплированием BRDF и русской рулеткой. Отсчёты
 * накапливаются в target: каждый вызов добавляет spp отсчётов.
 *
 *   spp      — отсчётов на пиксель в этом вызове;
 *   bounces  — максимум отражений;
 *   seed     — база генератора (разные кадры — разные seed);
 *   frame    — номер кадра (джиттер по Хальтону + seed);
 *   reset    — 1: очистить накопление перед рендером (новый кадр);
 *   max_radiance — ограничение яркости отсчёта (firefly clamp). Стандартная
 *                  практика трассировщиков: без него редкие яркие пути
 *                  дают одиночные «файрфлаи», из-за которых MSE-метрики
 *                  (PSNR) скачут на единицы дБ от кадра к кадру.
 *                  0 = не ограничивать.
 *
 * G-буфер заполняется при первом отсчёте и далее не меняется.
 * Возвращает 0 при успехе, -1 при ошибке.
 */
int rouge_render_cornell(RougeRenderTarget *t, const RougeCamera *cam, int spp,
                          int bounces, uint64_t seed, int frame, int reset,
                          float max_radiance);

/* Дисперсия оценки среднего: (E[x^2]-E[x]^2)/N — то, что SVGF называет
 * ошибкой фильтрации. Записывает w*h*3 float. */
int rouge_render_variance(const RougeRenderTarget *t, float *variance);

/* --- Денойзер ---

 * rouge_denoise_atrous: à-trous вейвлет-фильтр (5 отсчётов на ось, шаг
 * растёт: 1,2,4,8) с направляющими весами по нормали, относительной
 * глубине и яркости (последняя — по дисперсии оценки, как в SVGF).
 *
 *   iterations    — число проходов (0 = копия);
 *   phi_normal    — резкость по нормали;
 *   phi_depth     — резкость по относительной глубине;
 *   phi_variance  — резкость по яркости (в единицах ошибки).
 *
 * out должен быть того же размера, что color.
 */
int rouge_denoise_atrous(const RougeImage *color, const RougeGBuffer *g,
                         const float *variance, int iterations, float phi_normal,
                         float phi_depth, float phi_variance, RougeImage *out);

/* --- Темпоральный апскейлер ---

 * rouge_upsample_temporal: наш путь «как в DLSS, но без нейросети»:
 *   1) пространственная реконструкция 2x (Catmull-Rom);
 *   2) репроекция истории по векторам движения;
 *   3) клиппинг истории по окрестности текущего кадра (YCoCg AABB);
 *   4) проверка глубины (отбрасываем историю, если она с другой поверхности);
 *   5) смешивание с коэффициентом, падающим при большом движении.
 *
 *   low          — вход в низком разрешении (w x h);
 *   g            — G-буфер в низком разрешении;
 *   motion       — векторы движения в пикселях НИЗКОГО разрешения (w*h*2);
 *   history      — предыдущий кадр в ВЫСОКОМ разрешении (2w x 2h) или NULL;
 *   history_depth— глубина истории в высоком разрешении (2w*2h) или NULL;
 *   blend        — базовый коэффициент смешивания с историей (0..1);
 *   clip_gamma   — ширина коридора клиппинга в долях сигмы окрестности
 *                  (variance clipping; больше — мягче, меньше — жёстче);
 *   jitter_x/jitter_y — смещение отсчёта от центра пикселя в НИЗКОМ
 *                  разрешении (см. rouge_frame_jitter). Реконструкция с ним
 *                  сразу кладёт кадр в каноническую сетку;
 *   out          — результат в высоком разрешении (2w x 2h);
 *   new_history  — если не NULL, туда пишется результат (для след. кадра);
 *   new_history_depth — если не NULL, туда пишется глубина (2w*2h).
 */
int rouge_upsample_temporal(const RougeImage *low, const RougeGBuffer *g,
                            const float *motion, const RougeImage *history,
                            const float *history_depth, float blend,
                            float clip_gamma, float jitter_x, float jitter_y,
                            RougeImage *out, RougeImage *new_history,
                            float *new_history_depth);

/* Билинейный апскейлер 2x — база для сравнения в тестах. */
int rouge_upsample_bilinear(const RougeImage *low, RougeImage *out);

/* Джиттер кадра (смещение точки выборки внутри пикселя), каким его
 * использует рендер для первого отсчёта кадра frame при spp отсчётах.
 * Нужен вызывающему, чтобы при репроекции истории учесть, что кадр был
 * снят со сдвигом: без этого история берётся не с той точки и накопление
 * размазывается. */
void rouge_frame_jitter(int frame, int spp, float scale, float *jx, float *jy);

/* Векторы движения по движущейся камере: из мировых позиций поверхности.
 * motion — w*h*2 (пиксели низкого разрешения).
 *
 * Это ЧИСТОЕ движение камеры: проекция мировой точки от джиттера не
 * зависит, а джиттер компенсируется в реконструкции (см.
 * rouge_upsample_temporal). */
int rouge_motion_from_world(const RougeGBuffer *g, const RougeCamera *cam_cur,
                             const RougeCamera *cam_prev, float *motion);

/* --- Метрики --- */

/* PSNR по среднеквадратичному отклонению (max 1.0 -> 99 dB). */
double rouge_psnr(const RougeImage *a, const RougeImage *b);

/* SSIM по яркости, окно 11x11 гаусс sigma=1.5 (стандартная формула). */
double rouge_ssim(const RougeImage *a, const RougeImage *b);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  /* ROUGE_RENDER_H */
