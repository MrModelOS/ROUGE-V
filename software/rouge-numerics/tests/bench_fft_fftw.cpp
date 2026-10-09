// Наш FFT против FFTW (test-only зависимость, в библиотеку не входит).
//
// FFTW — эталонная открытая реализация; сравнение честное: та же машина,
// те же wall-clock (минимум из прогонов), корректность обязательна
// (допуск 1e-4 — single precision обеих сторон), скорость — только отчёт.
// Собирается только если найдены fftw3.h + libfftw3f, иначе этот тест
// отсутствует в сборке (а не «пропущен молча» — его просто нет).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <fftw3.h>

#include "rouge/numerics.h"

static int failures = 0;

#define EXPECT_TRUE(cond)                                        \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                \
    }                                                            \
  } while (0)

static double bench(int iters, void (*fn)(void *), void *ctx) {
  fn(ctx);
  double best = 1e300;
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fn(ctx);
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }
  return best;
}

struct Ctx {
  int n;
  const RougeComplex *in;
  RougeComplex *out;
};

static void run_rouge(void *p) {
  auto *c = (Ctx *)p;
  rouge_fft_f32(c->n, c->in, c->out, 0);
}

static fftwf_plan g_plan = nullptr;
static void run_fftw(void *p) {
  (void)p;
  fftwf_execute(g_plan);
}

int main() {
  std::mt19937 rng(0xF17u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int sizes[] = {64, 256, 1024, 4096, 100, 1000};
  for (int n : sizes) {
    std::vector<RougeComplex> in(n), got(n), want(n);
    for (auto &c : in) {
      c.re = dist(rng);
      c.im = dist(rng);
    }
    EXPECT_TRUE(rouge_fft_f32(n, in.data(), got.data(), 0) == 0);
    g_plan = fftwf_plan_dft_1d(n, (fftwf_complex *)in.data(),
                               (fftwf_complex *)want.data(), FFTW_FORWARD,
                               FFTW_ESTIMATE);
    EXPECT_TRUE(g_plan != nullptr);
    fftwf_execute(g_plan);
    double md = 0.0, sc = 1.0;
    for (int i = 0; i < n; ++i) {
      md = std::max(md, std::fabs((double)got[i].re - want[i].re));
      md = std::max(md, std::fabs((double)got[i].im - want[i].im));
      sc = std::max(sc, (double)std::fabs(want[i].re));
      sc = std::max(sc, (double)std::fabs(want[i].im));
    }
    if (md / sc > 1e-4) {
      std::printf("FAIL fft n=%d vs fftw: %g\n", n, md / sc);
      ++failures;
    }
    Ctx ctx{n, in.data(), got.data()};
    const double t_rouge = bench(7, run_rouge, &ctx);
    // FFTW замеряем тем же планом (in -> want).
    Ctx ctx2{n, in.data(), want.data()};
    (void)ctx2;
    const double t_fftw = bench(7, run_fftw, &ctx2);
    std::printf("fft n=%5d  rouge %9.3f ms  fftw %9.3f ms\n", n, t_rouge * 1e3,
                t_fftw * 1e3);
    fftwf_destroy_plan(g_plan);
    g_plan = nullptr;
  }
  if (failures == 0) std::printf("PASS: fft vs FFTW\n");
  return failures ? 1 : 0;
}
