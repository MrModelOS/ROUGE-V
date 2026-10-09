// Комплексный GEMM (c32/z64) против эталона в long double.
//
// Эталон — прямые петли в long double (другой порядок накопления, заведомо
// точнее обоих). Допуски: c32 — 1e-6 (f32-накопление четырёх GEMM),
// z64 — 1e-14.

#include <algorithm>
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

template <typename C>
static void ref_gemm(int M, int N, int K, const std::vector<C> &A,
                     const std::vector<C> &B, std::vector<C> &Out) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      long double rr = 0, ri = 0;
      for (int k = 0; k < K; ++k) {
        const long double ar = A[(size_t)i * K + k].re;
        const long double ai = A[(size_t)i * K + k].im;
        const long double br = B[(size_t)k * N + j].re;
        const long double bi = B[(size_t)k * N + j].im;
        rr += ar * br - ai * bi;
        ri += ar * bi + ai * br;
      }
      const size_t o = (size_t)i * N + j;
      Out[o].re = (decltype(Out[o].re))rr;
      Out[o].im = (decltype(Out[o].im))ri;
    }
}

int main() {
  std::mt19937 rng(0xC321u);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const int M = 17, N = 13, K = 23;  // некратные плитке
  std::vector<RougeComplex> A(M * K), B(K * N), gotC(M * N), wantC(M * N);
  std::vector<RougeComplexD> Ad(M * K), Bd(K * N), gotZ(M * N), wantZ(M * N);
  for (size_t i = 0; i < A.size(); ++i) {
    A[i].re = (float)dist(rng);
    A[i].im = (float)dist(rng);
    Ad[i].re = dist(rng);
    Ad[i].im = dist(rng);
  }
  for (size_t i = 0; i < B.size(); ++i) {
    B[i].re = (float)dist(rng);
    B[i].im = (float)dist(rng);
    Bd[i].re = dist(rng);
    Bd[i].im = dist(rng);
  }
  rouge_gemm_c32(M, N, K, A.data(), B.data(), gotC.data(), nullptr);
  ref_gemm(M, N, K, A, B, wantC);
  double md = 0.0, sc = 1.0;
  for (size_t i = 0; i < wantC.size(); ++i) {
    md = std::max({md, (double)std::fabs(gotC[i].re - wantC[i].re),
                   (double)std::fabs(gotC[i].im - wantC[i].im)});
    sc = std::max({sc, (double)std::fabs(wantC[i].re), (double)std::fabs(wantC[i].im)});
  }
  if (md / sc > 1e-6) {
    std::printf("c32 rel %g\n", md / sc);
    ++failures;
  }

  rouge_gemm_z64(M, N, K, Ad.data(), Bd.data(), gotZ.data(), nullptr);
  ref_gemm(M, N, K, Ad, Bd, wantZ);
  md = 0.0;
  sc = 1.0;
  for (size_t i = 0; i < wantZ.size(); ++i) {
    md = std::max({md, std::fabs(gotZ[i].re - wantZ[i].re),
                   std::fabs(gotZ[i].im - wantZ[i].im)});
    sc = std::max({sc, std::fabs(wantZ[i].re), std::fabs(wantZ[i].im)});
  }
  if (md / sc > 1e-14) {
    std::printf("z64 rel %g\n", md / sc);
    ++failures;
  }

  // NULL/нули без падения.
  rouge_gemm_c32(0, N, K, A.data(), B.data(), gotC.data(), nullptr);
  rouge_gemm_z64(M, N, K, nullptr, Bd.data(), gotZ.data(), nullptr);

  if (failures == 0) std::printf("PASS: complex c32/z64\n");
  return failures ? 1 : 0;
}
