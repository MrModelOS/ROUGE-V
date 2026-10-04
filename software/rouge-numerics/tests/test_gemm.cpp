// Тесты rouge-numerics: корректность каждого конфига против наивного эталона
// плюс отчёт о произведённой на этой машине скорости.
//
// Эталон — наивная тройная петля (naive triple loop). Никакого кэша и никаких
// «ожидаемых» чисел: расхождение считается, скорость печатается как есть.

#include <rouge/numerics.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

// Размеры из задания плюс несколько некратных плитке. 1x1x1 и 3x5x7 проверяют
// ветки «панель меньше плитки», 17x33x65 и 200x100x50 — края по всем трём
// измерениям, 128x128x128 и 256x192x128 — крупные панели.
struct Size {
  int m, n, k;
};

const Size kSizes[] = {
    {1, 1, 1},      {3, 5, 7},     {17, 33, 65},    {128, 128, 128},
    {200, 100, 50}, {64, 64, 128}, {129, 131, 127}, {256, 192, 128},
};

// Относительная допустимая погрешность из задания.
constexpr double kTol = 1e-5;

// Погрешность: max|C - ref| / max(||ref||_inf, 1). Единица в знаменателе
// защищает от деления на почти-нулевую сумму (при K=1 произведение может
// случайно оказаться близко к нулю), а не маскирует ошибку: значения
// элементов порядка единицы, поэтому масштаб здесь естественный.
double rel_error(const std::vector<float> &got, const std::vector<double> &ref) {
  double maxdiff = 0.0;
  double scale = 1.0;
  for (std::size_t t = 0; t < ref.size(); ++t) {
    maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(got[t]) - ref[t]));
    scale = std::max(scale, std::fabs(ref[t]));
  }
  return maxdiff / scale;
}

// Эталон: наивная тройная петля. Считаем в double, чтобы погрешность самого
// эталона не влияла на оценку нашего ядра.
std::vector<double> naive_ref(int M, int N, int K, const std::vector<float> &A,
                              const std::vector<float> &B) {
  std::vector<double> ref(static_cast<std::size_t>(M) * N, 0.0);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) {
        s += static_cast<double>(A[static_cast<std::size_t>(i) * K + k]) *
             static_cast<double>(B[static_cast<std::size_t>(k) * N + j]);
      }
      ref[static_cast<std::size_t>(i) * N + j] = s;
    }
  }
  return ref;
}

void fill_random(std::vector<float> &v, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float &x : v)
    x = dist(rng);
}

} // namespace

int main() {
  std::printf("rouge-numerics: ISA-вариант, которым считает этот процессор: %s\n",
              rouge_gemm_isa_variant());
  std::printf("конфигов-кандидатов у автотюнера: %d\n\n", rouge_gemm_candidate_count());

  const int n_cand = rouge_gemm_candidate_count();
  double worst_rel = 0.0;
  int failures = 0;

  for (const Size &sz : kSizes) {
    const int M = sz.m, N = sz.n, K = sz.k;
    std::mt19937 rng(static_cast<unsigned>(M * 7919 + N * 104729 + K));
    std::vector<float> A(static_cast<std::size_t>(M) * K);
    std::vector<float> B(static_cast<std::size_t>(K) * N);
    fill_random(A, rng);
    fill_random(B, rng);
    const std::vector<double> ref = naive_ref(M, N, K, A, B);

    std::printf("--- M=%d N=%d K=%d (элементов C: %lld)\n", M, N, K,
                static_cast<long long>(static_cast<long long>(M) * N));

    // 1. Корректность КАЖДОГО кандидата: сравнение с наивным эталоном.
    for (int ci = 0; ci < n_cand; ++ci) {
      const RougeGemmConfig cfg = rouge_gemm_candidate(ci);
      std::vector<float> C(static_cast<std::size_t>(M) * N, -7.0f); // мусор на входе
      rouge_gemm_f32(M, N, K, A.data(), B.data(), C.data(), &cfg);
      const double rel = rel_error(C, ref);
      worst_rel = std::max(worst_rel, rel);
      const bool ok = std::isfinite(rel) && rel <= kTol;
      if (!ok)
        ++failures;
      std::printf("    cand %d tm=%-3d tn=%-3d tk=%-3d unroll=%d split_k=%d  "
                  "rel_err=%.3e  %s\n",
                  ci, cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.unroll, cfg.split_k, rel,
                  ok ? "ok" : "FAIL");
    }

    // 2. Конфиг по умолчанию (cfg == nullptr) — тоже проверяем отдельно.
    {
      std::vector<float> C(static_cast<std::size_t>(M) * N, -7.0f);
      rouge_gemm_f32(M, N, K, A.data(), B.data(), C.data(), nullptr);
      const double rel = rel_error(C, ref);
      worst_rel = std::max(worst_rel, rel);
      const bool ok = std::isfinite(rel) && rel <= kTol;
      if (!ok)
        ++failures;
      std::printf("    конфиг по умолчанию (cfg=NULL)               rel_err=%.3e  %s\n", rel,
                  ok ? "ok" : "FAIL");
    }

    // 3. Автотюнер: результат обязан совпасть с эталоном, профиль — непустой.
    std::vector<float> C(static_cast<std::size_t>(M) * N, -7.0f);
    const RougeGemmProfile prof = rouge_gemm_tuned(M, N, K, A.data(), B.data(), C.data());
    const double rel = rel_error(C, ref);
    worst_rel = std::max(worst_rel, rel);
    const bool ok = std::isfinite(rel) && rel <= kTol && prof.gflops > 0.0 && prof.iters > 0;
    if (!ok)
      ++failures;
    std::printf("    автотюнер: tm=%d tn=%d tk=%d unroll=%d split_k=%d  rel_err=%.3e  "
                "замеров=%lld  GFLOP/s=%.2f (замерено здесь и сейчас)\n",
                prof.cfg.tile_m, prof.cfg.tile_n, prof.cfg.tile_k, prof.cfg.unroll,
                prof.cfg.split_k, rel, prof.iters, prof.gflops);

    // 4. Отдельный замер скорости выбранного автотюнером конфига: это тот же
    //    путь, которым пользуется rouge_gemm_flops_per_sec.
    const double gflops =
        rouge_gemm_flops_per_sec(M, N, K, A.data(), B.data(), C.data(), &prof.cfg);
    std::printf("    скорость выбранного конфига: %.2f GFLOP/s\n\n", gflops);
  }

  std::printf("итого: худшая относительная погрешность по всем конфигам и размерам = %.3e "
              "(допуск %.0e)\n",
              worst_rel, kTol);
  if (failures != 0) {
    std::printf("FAIL: расхождений с эталоном или пустых профилей: %d\n", failures);
    return 1;
  }
  std::printf("PASS: все конфиги совпали с наивным эталоном; скорости выше замерены на этой "
              "машине\n");
  return 0;
}