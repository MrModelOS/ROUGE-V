// FP64 GEMM: тот же блочный алгоритм в double против наивного эталона.
//
// Каждый конфиг-кандидат (как в test_gemm.cpp для f32) на размерах с краями,
// допуск 1e-12 относительно шкалы — double-накопление честно точнее.
// Плюс split_k многопоточно и NULL/нули без падения.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "rouge/numerics.h"

static int failures = 0;

#define EXPECT_TRUE(cond)                                        \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                \
    }                                                            \
  } while (0)

static void naive(int M, int N, int K, const double *A, const double *B,
                  double *C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      long double s = 0;
      for (int k = 0; k < K; ++k)
        s += (long double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
      C[(size_t)i * N + j] = (double)s;
    }
}

int main() {
  std::mt19937 rng(0xD6464u);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const int sizes[][3] = {{1, 1, 1}, {3, 5, 7}, {17, 33, 65}, {64, 64, 128}};
  const int ncand = rouge_gemm_candidate_count();
  EXPECT_TRUE(ncand > 0);
  double worst = 0.0;
  for (int c = 0; c < ncand; ++c) {
    const RougeGemmConfig cfg = rouge_gemm_candidate(c);
    for (auto &sz : sizes) {
      const int M = sz[0], N = sz[1], K = sz[2];
      std::vector<double> A((size_t)M * K), B((size_t)K * N), got((size_t)M * N),
          want(got.size());
      for (auto &v : A) v = dist(rng);
      for (auto &v : B) v = dist(rng);
      rouge_gemm_f64(M, N, K, A.data(), B.data(), got.data(), &cfg);
      naive(M, N, K, A.data(), B.data(), want.data());
      double md = 0.0, sc = 1.0;
      for (size_t i = 0; i < want.size(); ++i) {
        md = std::max(md, std::fabs(got[i] - want[i]));
        sc = std::max(sc, std::fabs(want[i]));
      }
      worst = std::max(worst, md / sc);
      if (md / sc > 1e-12) {
        std::printf("f64 cand %d %dx%dx%d rel %g\n", c, M, N, K, md / sc);
        ++failures;
      }
    }
  }
  // split_k многопоточно.
  {
    const int M = 48, N = 40, K = 64;
    std::vector<double> A((size_t)M * K), B((size_t)K * N), got((size_t)M * N),
        want(got.size());
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    RougeGemmConfig cfg = rouge_gemm_default_config();
    cfg.split_k = 4;
    rouge_gemm_f64(M, N, K, A.data(), B.data(), got.data(), &cfg);
    naive(M, N, K, A.data(), B.data(), want.data());
    double md = 0.0, sc = 1.0;
    for (size_t i = 0; i < want.size(); ++i) {
      md = std::max(md, std::fabs(got[i] - want[i]));
      sc = std::max(sc, std::fabs(want[i]));
    }
    EXPECT_TRUE(md / sc < 1e-12);
  }
  // NULL/нули.
  {
    double x = 1.0;
    rouge_gemm_f64(0, 4, 4, &x, &x, &x, nullptr);
    rouge_gemm_f64(4, 4, 4, nullptr, &x, &x, nullptr);
    EXPECT_TRUE(x == 1.0);
  }
  if (failures == 0)
    std::printf("PASS: gemm f64 (worst rel %g, isa %s)\n", worst,
                rouge_gemm_isa_variant());
  return failures ? 1 : 0;
}
