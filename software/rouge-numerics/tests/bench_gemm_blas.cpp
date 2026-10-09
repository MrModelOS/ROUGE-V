// Замер Wave 2: наш GEMM против системного BLAS и наивного эталона.
//
// Честность прежде всего: это НЕ сравнение с cuBLAS на GPU и НЕ заявление
// «быстрее OpenBLAS». Здесь CPU-библиотека меряется против CPU-библиотеки на
// той же машине, теми же wall-clock. Что линкуется — печатается:
//   * если найден cblas.h + libblas/libcblas — меряем cblas_sgemm;
//   * иначе — только наивная тройная петля как пол.
// Корректность обязательна (допуск 1e-5 против cblas/naive), скорость —
// только отчёт, тест не падает на «медленнее».

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "rouge/numerics.h"

#if defined(ROUGE_HAVE_CBLAS)
#include <cblas.h>
#endif

static void naive(int M, int N, int K, const float *A, const float *B,
                  float *C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0;
      for (int k = 0; k < K; ++k) s += (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
      C[(size_t)i * N + j] = (float)s;
    }
}

static double bench(int iters, void (*fn)(void *), void *ctx) {
  fn(ctx);  // разогрев
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
  int M, N, K;
  const float *A, *B;
  float *C;
};

static void naive_d(int M, int N, int K, const double *A, const double *B,
                    double *C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      long double s = 0;
      for (int k = 0; k < K; ++k) s += (long double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
      C[(size_t)i * N + j] = (double)s;
    }
}

struct CtxD {
  int M, N, K;
  const double *A, *B;
  double *C;
};

static void run_rouge_d(void *p) {
  auto *c = (CtxD *)p;
  rouge_gemm_f64(c->M, c->N, c->K, c->A, c->B, c->C, nullptr);
}
static void run_naive_d(void *p) {
  auto *c = (CtxD *)p;
  naive_d(c->M, c->N, c->K, c->A, c->B, c->C);
}
#if defined(ROUGE_HAVE_CBLAS)
static void run_cblas_d(void *p) {
  auto *c = (CtxD *)p;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c->M, c->N, c->K, 1.0,
              c->A, c->K, c->B, c->N, 0.0, c->C, c->N);
}
#endif

static void run_rouge(void *p) {
  auto *c = (Ctx *)p;
  rouge_gemm_f32(c->M, c->N, c->K, c->A, c->B, c->C, nullptr);
}
static void run_naive(void *p) {
  auto *c = (Ctx *)p;
  naive(c->M, c->N, c->K, c->A, c->B, c->C);
}
#if defined(ROUGE_HAVE_CBLAS)
static void run_cblas(void *p) {
  auto *c = (Ctx *)p;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c->M, c->N, c->K,
              1.0f, c->A, c->K, c->B, c->N, 0.0f, c->C, c->N);
}
#endif

int main() {
  std::mt19937 rng(0xBE4Cu);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int sizes[][3] = {{64, 64, 64}, {128, 128, 128}, {256, 192, 128}};
  int failures = 0;
  std::printf("isa=%s %s\n", rouge_gemm_isa_variant(),
#if defined(ROUGE_HAVE_CBLAS)
              "blas=cblas(system)"
#else
              "blas=naive-only(no cblas at build)"
#endif
  );
  for (auto &sz : sizes) {
    const int M = sz[0], N = sz[1], K = sz[2];
    std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N),
        ref(C.size());
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    Ctx ctx{M, N, K, A.data(), B.data(), C.data()};
    const double flops = 2.0 * M * N * K;
    const double t_rouge = bench(5, run_rouge, &ctx);
    std::vector<float> got = C;
    const double t_naive = bench(3, run_naive, &ctx);
    ref = C;
    double md = 0, sc = 1;
    for (size_t i = 0; i < ref.size(); ++i) {
      md = std::max(md, std::fabs((double)got[i] - ref[i]));
      sc = std::max(sc, std::fabs((double)ref[i]));
    }
    if (md / sc > 1e-5) {
      std::printf("FAIL %dx%dx%d vs naive: %g\n", M, N, K, md / sc);
      ++failures;
    }
#if defined(ROUGE_HAVE_CBLAS)
    const double t_cblas = bench(5, run_cblas, &ctx);
    std::vector<float> bc = C;
    md = 0;
    sc = 1;
    for (size_t i = 0; i < ref.size(); ++i) {
      md = std::max(md, std::fabs((double)got[i] - bc[i]));
      sc = std::max(sc, std::fabs((double)bc[i]));
    }
    if (md / sc > 1e-5) {
      std::printf("FAIL %dx%dx%d vs cblas: %g\n", M, N, K, md / sc);
      ++failures;
    }
    std::printf("%4dx%4dx%4d  rouge %7.3f GFLOP/s  cblas %7.3f GFLOP/s  naive %7.3f GFLOP/s\n",
                M, N, K, flops / t_rouge / 1e9, flops / t_cblas / 1e9,
                flops / t_naive / 1e9);
#else
    std::printf("%4dx%4dx%4d  rouge %7.3f GFLOP/s  naive %7.3f GFLOP/s\n", M,
                N, K, flops / t_rouge / 1e9, flops / t_naive / 1e9);
#endif
  }
  // --- FP64: свой f64-GEMM против dgemm/наивного ---
  {
    const int M = 128, N = 128, K = 128;
    std::uniform_real_distribution<double> ddist(-1.0, 1.0);
    std::vector<double> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
    for (auto &v : A) v = ddist(rng);
    for (auto &v : B) v = ddist(rng);
    CtxD ctx{M, N, K, A.data(), B.data(), C.data()};
    const double flops = 2.0 * M * N * K;
    const double t_rouge = bench(5, run_rouge_d, &ctx);
    std::vector<double> got = C;
    const double t_naive = bench(3, run_naive_d, &ctx);
    std::vector<double> ref = C;
    double md = 0, sc = 1;
    for (size_t i = 0; i < ref.size(); ++i) {
      md = std::max(md, std::fabs(got[i] - ref[i]));
      sc = std::max(sc, std::fabs(ref[i]));
    }
    if (md / sc > 1e-12) {
      std::printf("FAIL f64 vs naive: %g\n", md / sc);
      ++failures;
    }
#if defined(ROUGE_HAVE_CBLAS)
    const double t_cblas = bench(5, run_cblas_d, &ctx);
    std::vector<double> bc = C;
    md = 0;
    sc = 1;
    for (size_t i = 0; i < ref.size(); ++i) {
      md = std::max(md, std::fabs(got[i] - bc[i]));
      sc = std::max(sc, std::fabs(bc[i]));
    }
    if (md / sc > 1e-12) {
      std::printf("FAIL f64 vs cblas: %g\n", md / sc);
      ++failures;
    }
    std::printf("f64 %dx%dx%d  rouge %7.3f GFLOP/s  cblas %7.3f GFLOP/s  naive %7.3f GFLOP/s\n",
                M, N, K, flops / t_rouge / 1e9, flops / t_cblas / 1e9,
                flops / t_naive / 1e9);
#else
    std::printf("f64 %dx%dx%d  rouge %7.3f GFLOP/s  naive %7.3f GFLOP/s\n", M,
                N, K, flops / t_rouge / 1e9, flops / t_naive / 1e9);
#endif
  }
  if (failures == 0) std::printf("PASS: bench (correctness vs baseline)\n");
  return failures ? 1 : 0;
}
