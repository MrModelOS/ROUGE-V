/*
 * Path tracer: Cornell-box (диффузная сцена с площадным источником).
 *
 * Почему так: чтобы измерять денойзер честно, нужен ЭТАЛОН — тот же рендер,
 * посчитанный со многими отсчётами. Для этого рендер должен быть
 * несмещённым, иначе «улучшение» PSNR может быть артефактом смещения.
 *
 * Что реализовано:
 *   * next-event estimation (прямой свет сэмплируется на площадке);
 *   * MIS с power heuristic (beta=2) между NEE и BRDF-сэмплированием —
 *     без него NEE-свет считался бы дважды (смещение вверх);
 *   * косинусное сэмплирование диффузной BRDF;
 *   * русская рулетка после 3-го отскока;
 *   * накопление по отсчётам (sum, sumsq) — из этого получается дисперсия
 *     оценки, которая нужна денойзеру (SVGF-подход).
 *
 * Сцена описана в коде: box [-1,1]x[0,2]x[-1,1], белая коробка, красная
 * левая стена, зелёная правая, белая задняя, источник — квадрат на потолке.
 * Всё это — наши собственные данные, никаких чужих ассетов.
 */

#include "rouge/render.h"

#include "rouge/numerics.h"  // наш splitmix64 (тот же, что во всём стеке)

#include <cmath>
#include <cstring>
#include <vector>

namespace {

// --------------------------------------------------------------------- math
struct Vec3 {
  float x, y, z;
};

inline Vec3 v3(float x, float y, float z) { return Vec3{x, y, z}; }
inline Vec3 operator+(const Vec3 &a, const Vec3 &b) {
  return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline Vec3 operator-(const Vec3 &a, const Vec3 &b) {
  return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline Vec3 operator*(const Vec3 &a, float s) {
  return v3(a.x * s, a.y * s, a.z * s);
}
inline Vec3 operator*(const Vec3 &a, const Vec3 &b) {
  return v3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline float dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline Vec3 normalize(const Vec3 &a) {
  const float l = std::sqrt(dot(a, a));
  return l > 0.0f ? a * (1.0f / l) : a;
}
inline float maxc(const Vec3 &a) {
  return a.x > a.y ? (a.x > a.z ? a.x : a.z) : (a.y > a.z ? a.y : a.z);
}

// ------------------------------------------------------------------- сцена
constexpr float kBoxMinX = -1.0f, kBoxMaxX = 1.0f;
constexpr float kBoxMinY = 0.0f, kBoxMaxY = 2.0f;
constexpr float kBoxMinZ = -1.0f, kBoxMaxZ = 1.0f;

constexpr float kLightY = 1.99f;      // чуть ниже потолка
constexpr float kLightHalfX = 0.25f;   // половина стороны источника по X
constexpr float kLightHalfZ = 0.25f;
constexpr float kLightArea = 4.0f * kLightHalfX * kLightHalfZ;

const Vec3 kLightNormal = v3(0.0f, -1.0f, 0.0f);
const Vec3 kLightRadiance = v3(30.0f, 30.0f, 30.0f);

const Vec3 kWallWhite = v3(0.73f, 0.73f, 0.73f);
const Vec3 kWallRed = v3(0.65f, 0.05f, 0.05f);
const Vec3 kWallGreen = v3(0.12f, 0.45f, 0.15f);

// Плоскости стен: (нормаль внутрь, точка на плоскости, альбедо).
struct Wall {
  Vec3 n;
  float d;      // n . p = d
  Vec3 albedo;
};

inline Wall make_wall(Vec3 n, float d, Vec3 albedo) {
  return Wall{n, d, albedo};
}

inline void build_walls(Wall walls[5]) {
  walls[0] = make_wall(v3(0.0f, 1.0f, 0.0f), kBoxMinY, kWallWhite);   // пол
  walls[1] = make_wall(v3(0.0f, -1.0f, 0.0f), -kBoxMaxY, kWallWhite);  // потолок
  walls[2] = make_wall(v3(1.0f, 0.0f, 0.0f), -kBoxMinX, kWallRed);     // левая
  walls[3] = make_wall(v3(-1.0f, 0.0f, 0.0f), kBoxMaxX, kWallGreen);   // правая
  walls[4] = make_wall(v3(0.0f, 0.0f, -1.0f), -kBoxMaxZ, kWallWhite);  // задняя
}

// Пересечение луча с квадратом источника (плоскость y = kLightY).
inline bool hit_light(const Vec3 &o, const Vec3 &d, float tmax, float *t_out) {
  if (std::fabs(d.y) < 1e-8f) return false;
  const float t = (kLightY - o.y) / d.y;
  if (t <= 1e-4f || t >= tmax) return false;
  const Vec3 p = o + d * t;
  if (std::fabs(p.x) > kLightHalfX) return false;
  if (std::fabs(p.z) > kLightHalfZ) return false;
  *t_out = t;
  return true;
}

// Ближайшая стена. Возвращает false, если луч ни во что не попал.
inline bool trace_walls(const Wall walls[5], const Vec3 &o, const Vec3 &d,
                        float *t_out, int *idx_out) {
  float best = 1e30f;
  int best_idx = -1;
  for (int i = 0; i < 5; ++i) {
    const float denom = dot(walls[i].n, d);
    if (denom > -1e-8f && denom < 1e-8f) continue;
    const float t = (walls[i].d - dot(walls[i].n, o)) / denom;
    if (t > 1e-4f && t < best) {
      best = t;
      best_idx = i;
    }
  }
  if (best_idx < 0) return false;
  *t_out = best;
  *idx_out = best_idx;
  return true;
}

// Пересечение со светом + со стенами; свет проверяем отдельно, потому что он
// лежит ВНУТРИ коробки (чуть ниже потолка).
struct Hit {
  float t;
  bool is_light;
  int wall;
};

inline bool trace_scene(const Wall walls[5], const Vec3 &o, const Vec3 &d,
                        Hit *hit) {
  float tw;
  int wi;
  const bool wall_hit = trace_walls(walls, o, d, &tw, &wi);
  float tl;
  const bool light_hit = hit_light(o, d, wall_hit ? tw : 1e30f, &tl);
  if (light_hit) {
    hit->t = tl;
    hit->is_light = true;
    hit->wall = -1;
    return true;
  }
  if (wall_hit) {
    hit->t = tw;
    hit->is_light = false;
    hit->wall = wi;
    return true;
  }
  return false;
}

// -------------------------------------------------------------- сэмплирование

// Косинусное сэмплирование вокруг нормали n (диффузная BRDF).
inline Vec3 sample_cosine_hemisphere(const Vec3 &n, RougeRng *rng) {
  const float u1 = rouge_rng_uniform_f32(rng);
  const float u2 = rouge_rng_uniform_f32(rng);
  const float r = std::sqrt(u1);
  const float phi = 6.28318530718f * u2;
  const float x = r * std::cos(phi);
  const float y = r * std::sin(phi);
  const float z = std::sqrt(std::fmax(0.0f, 1.0f - u1));
  // Строим кадр (n, t, b).
  const Vec3 helper = (std::fabs(n.x) > 0.57735f) ? v3(1.0f, 0.0f, 0.0f)
                                                  : v3(0.0f, 1.0f, 0.0f);
  const Vec3 t = normalize(cross(helper, n));
  const Vec3 b = cross(n, t);
  return normalize(t * x + b * y + n * z);
}

// Хальтон: джиттер внутри пикселя по двум основаниям.
inline float halton(int index, int base) {
  float f = 1.0f;
  float r = 0.0f;
  int i = index + 1;
  while (i > 0) {
    f /= (float)base;
    r += f * (float)(i % base);
    i /= base;
  }
  return r;
}

// ----------------------------------------------------------------- камера
struct CameraBasis {
  Vec3 forward, right, up;
  float tan_half, aspect;
};

inline CameraBasis make_basis(const RougeCamera &cam, int w, int h) {
  CameraBasis b;
  b.forward = normalize(v3(cam.target[0] - cam.pos[0], cam.target[1] - cam.pos[1],
                            cam.target[2] - cam.pos[2]));
  const Vec3 up_hint = v3(cam.up[0], cam.up[1], cam.up[2]);
  b.right = normalize(cross(b.forward, up_hint));
  b.up = cross(b.right, b.forward);
  b.tan_half = std::tan(0.5f * cam.fov_y * 3.14159265358979f / 180.0f);
  b.aspect = (float)w / (float)h;
  return b;
}

// Проецируем мировую точку в пиксельные координаты (соглашение [i, i+1):
// пиксель i занимает [i, i+1), центр в i+0.5).
// Джиттер здесь НЕ вычитается: проекция мировой точки от джиттера не
// зависит, а разность проекций даёт чистое движение камеры. Джиттер
// компенсируется отдельно — в реконструкции (jitter-aware).
// Возвращает false, если точка за камерой.
inline bool project_point(const RougeCamera &cam, int w, int h, const Vec3 &p,
                          float *px, float *py) {
  const CameraBasis b = make_basis(cam, w, h);
  const Vec3 rel = p - v3(cam.pos[0], cam.pos[1], cam.pos[2]);
  const float zc = dot(rel, b.forward);
  if (zc <= 1e-4f) return false;
  const float xc = dot(rel, b.right);
  const float yc = dot(rel, b.up);
  *px = (xc / (zc * b.tan_half * b.aspect) + 1.0f) * 0.5f * (float)w;
  *py = (1.0f - (yc / (zc * b.tan_half))) * 0.5f * (float)h;
  return true;
}

}  // namespace

extern "C" {

// ------------------------------------------------------------------- время
RougeImage *rouge_image_new(int w, int h) {
  if (w <= 0 || h <= 0) return nullptr;
  RougeImage *img = new RougeImage();
  img->w = w;
  img->h = h;
  img->px = new float[(size_t)w * h * 3]();
  return img;
}

void rouge_image_free(RougeImage *img) {
  if (!img) return;
  delete[] img->px;
  delete img;
}

RougeRenderTarget *rouge_target_new(int w, int h) {
  if (w <= 0 || h <= 0) return nullptr;
  RougeRenderTarget *t = new RougeRenderTarget();
  t->w = w;
  t->h = h;
  t->samples = 0;
  const size_t n = (size_t)w * h;
  t->sum = new float[n * 3]();
  t->sumsq = new float[n * 3]();
  t->g.w = w;
  t->g.h = h;
  t->g.normal = new float[n * 3]();
  t->g.depth = new float[n]();
  t->g.albedo = new float[n * 3]();
  t->g.position = new float[n * 3]();
  return t;
}

void rouge_target_free(RougeRenderTarget *t) {
  if (!t) return;
  delete[] t->sum;
  delete[] t->sumsq;
  delete[] t->g.normal;
  delete[] t->g.depth;
  delete[] t->g.albedo;
  delete[] t->g.position;
  delete t;
}

void rouge_camera_default(RougeCamera *cam) {
  if (!cam) return;
  cam->pos[0] = 0.0f;
  cam->pos[1] = 1.0f;
  cam->pos[2] = -3.2f;
  cam->target[0] = 0.0f;
  cam->target[1] = 1.0f;
  cam->target[2] = 0.0f;
  cam->up[0] = 0.0f;
  cam->up[1] = 1.0f;
  cam->up[2] = 0.0f;
  cam->fov_y = 40.0f;
  cam->jitter_scale = 1.0f;
}

// ------------------------------------------------------------------ рендер
int rouge_render_cornell(RougeRenderTarget *t, const RougeCamera *cam, int spp,
                          int bounces, uint64_t seed, int frame, int reset,
                          float max_radiance) {
  if (!t || !cam || spp <= 0 || bounces <= 0) return -1;
  const int w = t->w, h = t->h;
  if (reset) {
    std::memset(t->sum, 0, sizeof(float) * (size_t)w * h * 3);
    std::memset(t->sumsq, 0, sizeof(float) * (size_t)w * h * 3);
    t->samples = 0;
  }
  Wall walls[5];
  build_walls(walls);
  const CameraBasis basis = make_basis(*cam, w, h);
  const float inv_h = 1.0f / (float)h;
  const float inv_w = 1.0f / (float)w;

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      RougeRng rng;
      rouge_rng_seed(&rng, seed ^ (uint64_t)(frame * 0x9E3779B97F4A7C15ull));
      // Отсчёты внутри пикселя разводим по индексу (кадр, сэмпл).
      double acc[3] = {0.0, 0.0, 0.0};
      double accsq[3] = {0.0, 0.0, 0.0};
      bool gbuf_set = false;

      for (int s = 0; s < spp; ++s) {
        const int hidx = frame * spp + s;
        // Джиттер — смещение ОТ ЦЕНТРА пикселя: при scale=0 отсчёт берётся
        // ровно в центре (x+0.5), иначе картинка систематически смещалась
        // бы на полпикселя относительно проекции и реконструкции.
        const float jx = cam->jitter_scale * (halton(hidx, 2) - 0.5f);
        const float jy = cam->jitter_scale * (halton(hidx, 3) - 0.5f);
        const float sx = ((float)x + 0.5f + jx) * inv_w;
        const float sy = ((float)y + 0.5f + jy) * inv_h;
        const float px = (2.0f * sx - 1.0f) * basis.aspect * basis.tan_half;
        const float py = (1.0f - 2.0f * sy) * basis.tan_half;
        Vec3 dir = normalize(basis.forward + basis.right * px +
                             basis.up * py);
        const Vec3 org = v3(cam->pos[0], cam->pos[1], cam->pos[2]);

        Vec3 radiance = v3(0.0f, 0.0f, 0.0f);
        Vec3 throughput = v3(1.0f, 1.0f, 1.0f);
        Vec3 p = org;
        Vec3 n = basis.forward;
        Vec3 albedo = v3(0.0f, 0.0f, 0.0f);
        float depth = 0.0f;
        bool first = true;

        for (int bounce = 0; bounce < bounces; ++bounce) {
          Hit hit;
          if (!trace_scene(walls, p, dir, &hit)) break;

          if (hit.is_light) {
            // Путь пришёл в источник: вклад от BRDF-сэмплирования, с весом
            // MIS против NEE.
            const float cos_light = std::fabs(dot(kLightNormal, dir));
            const float cos_surface = std::fabs(dot(n, dir));
            if (cos_light > 0.0f && cos_surface > 0.0f) {
              const float pdf_bsdf = cos_surface / 3.14159265358979f;
              const float d2 = hit.t * hit.t;
              const float pdf_light = d2 / (cos_light * kLightArea);
              const float w = pdf_bsdf * pdf_bsdf /
                              (pdf_bsdf * pdf_bsdf + pdf_light * pdf_light);
              const Vec3 f = albedo * (1.0f / 3.14159265358979f);
              const Vec3 contrib = f * kLightRadiance * (cos_light * w / pdf_bsdf);
              radiance = radiance + throughput * contrib;
            }
            break;  // источник — конец пути
          }

          const Vec3 hitp = p + dir * hit.t;
          const Vec3 sn = walls[hit.wall].n;
          const Vec3 salb = walls[hit.wall].albedo;
          if (first) {
            n = sn;
            albedo = salb;
            depth = hit.t;
            // Мировая позиция для G-буфера.
            const size_t gi = ((size_t)y * w + x) * 3;
            t->g.position[gi + 0] = hitp.x;
            t->g.position[gi + 1] = hitp.y;
            t->g.position[gi + 2] = hitp.z;
            t->g.normal[gi + 0] = sn.x;
            t->g.normal[gi + 1] = sn.y;
            t->g.normal[gi + 2] = sn.z;
            t->g.albedo[gi + 0] = salb.x;
            t->g.albedo[gi + 1] = salb.y;
            t->g.albedo[gi + 2] = salb.z;
            t->g.depth[(size_t)y * w + x] = depth;
            gbuf_set = true;
            first = false;
          }

          // NEE: прямой свет.
          {
            const float u = rouge_rng_uniform_f32(&rng);
            const float v = rouge_rng_uniform_f32(&rng);
            const Vec3 lp = v3((u - 0.5f) * 2.0f * kLightHalfX, kLightY,
                                (v - 0.5f) * 2.0f * kLightHalfZ);
            const Vec3 to_l = lp - hitp;
            const float d2 = dot(to_l, to_l);
            const float dist = std::sqrt(d2);
            if (dist > 1e-5f) {
              const Vec3 wi = to_l * (1.0f / dist);
              const float cos_s = dot(sn, wi);
              const float cos_l = -dot(kLightNormal, wi);
              if (cos_s > 0.0f && cos_l > 0.0f) {
                // Проверка затенения: луч до источника не должен попасть в стену.
                Hit sh;
                bool occluded = false;
                if (trace_scene(walls, hitp, wi, &sh) && sh.t < dist - 1e-3f &&
                    !sh.is_light)
                  occluded = true;
                if (!occluded) {
                  const float pdf_light = d2 / (cos_l * kLightArea);
                  const float pdf_bsdf = cos_s / 3.14159265358979f;
                  const float w = pdf_light * pdf_light /
                                  (pdf_light * pdf_light + pdf_bsdf * pdf_bsdf);
                  const Vec3 f = salb * (1.0f / 3.14159265358979f);
                  const Vec3 contrib = f * kLightRadiance * (cos_s * w / pdf_light);
                  radiance = radiance + throughput * contrib;
                }
              }
            }
          }

          // Русская рулетка после 3-го отскока.
          if (bounce >= 2) {
            const float q = maxc(throughput);
            if (q < 0.05f) break;
            if (rouge_rng_uniform_f32(&rng) > q) break;
            throughput = throughput * (1.0f / q);
          }

          // Следующий отскок: косинусное сэмплирование вокруг нормали.
          const Vec3 wi = sample_cosine_hemisphere(sn, &rng);
          const float cos_s = dot(sn, wi);
          if (cos_s <= 0.0f) break;
          throughput = throughput * salb;  // f * cos / pdf = albedo (диффузная)
          p = hitp;
          dir = wi;
          n = sn;
        }

        // Firefly clamp: ограничиваем яркость отсчёта. Стандартная
        // практика (см. «clamp indirect radiance»), без неё одиночные
        // яркие пути делают PSNR нестабильным.
        if (max_radiance > 0.0f) {
          radiance.x = radiance.x > max_radiance ? max_radiance : radiance.x;
          radiance.y = radiance.y > max_radiance ? max_radiance : radiance.y;
          radiance.z = radiance.z > max_radiance ? max_radiance : radiance.z;
        }

        acc[0] += radiance.x;
        acc[1] += radiance.y;
        acc[2] += radiance.z;
        accsq[0] += (double)radiance.x * radiance.x;
        accsq[1] += (double)radiance.y * radiance.y;
        accsq[2] += (double)radiance.z * radiance.z;
        (void)gbuf_set;
      }

      const size_t pi = ((size_t)y * w + x) * 3;
      for (int c = 0; c < 3; ++c) {
        t->sum[pi + c] += (float)acc[c];
        t->sumsq[pi + c] += (float)accsq[c];
      }
    }
  }
  t->samples += spp;
  return 0;
}

void rouge_frame_jitter(int frame, int spp, float scale, float *jx, float *jy) {
  if (!jx || !jy) return;
  const int hidx = frame * (spp > 0 ? spp : 1);
  // Смещение от центра пикселя (то же, что использует рендер).
  *jx = scale * (halton(hidx, 2) - 0.5f);
  *jy = scale * (halton(hidx, 3) - 0.5f);
}

int rouge_render_variance(const RougeRenderTarget *t, float *variance) {  if (!t || !variance || t->samples <= 0) return -1;
  const size_t n = (size_t)t->w * t->h * 3;
  const double inv_n = 1.0 / (double)t->samples;
  for (size_t i = 0; i < n; ++i) {
    const double mean = (double)t->sum[i] * inv_n;
    const double m2 = (double)t->sumsq[i] * inv_n;
    double v = m2 - mean * mean;
    if (v < 0.0) v = 0.0;
    // Дисперсия оценки среднего: Var/N.
    variance[i] = (float)(v * inv_n);
  }
  return 0;
}

int rouge_motion_from_world(const RougeGBuffer *g, const RougeCamera *cam_cur,
                             const RougeCamera *cam_prev, float *motion) {
  if (!g || !cam_cur || !cam_prev || !motion) return -1;
  for (int y = 0; y < g->h; ++y) {
    for (int x = 0; x < g->w; ++x) {
      const size_t gi = ((size_t)y * g->w + x) * 3;
      const Vec3 wp = v3(g->position[gi], g->position[gi + 1], g->position[gi + 2]);
      float cx = 0, cy = 0, px = 0, py = 0;
      const bool ok = project_point(*cam_cur, g->w, g->h, wp, &cx, &cy);
      const bool okp = project_point(*cam_prev, g->w, g->h, wp, &px, &py);
      if (!ok || !okp) {
        motion[((size_t)y * g->w + x) * 2 + 0] = 0.0f;
        motion[((size_t)y * g->w + x) * 2 + 1] = 0.0f;
        continue;
      }
      motion[((size_t)y * g->w + x) * 2 + 0] = cx - px;
      motion[((size_t)y * g->w + x) * 2 + 1] = cy - py;
    }
  }
  return 0;
}

}  // extern "C"
