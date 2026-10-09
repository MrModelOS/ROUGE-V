/*
 * Комплексный GEMM Wave 2: C[M][N] = A[M][K] * B[K][N] в комплексной
 * арифметике, через четыре вещественных GEMM своего ядра:
 *   Cre = Are*Bre - Aim*Bim,  Cim = Are*Bim + Aim*Bre.
 * Никакого отдельного комплексного микроядра: переиспользование
 * протестированного вещественного пути целиком (упаковка, автотюнер через
 * cfg, split_k). Цена — четыре прохода по K вместо одного fused; это
 * задокументированный размен, не скрытый.
 */

#include "rouge/numerics.h"

#include <cstddef>
#include <vector>

extern "C" {

void rouge_gemm_c32(int M, int N, int K, const RougeComplex *A,
                    const RougeComplex *B, RougeComplex *C,
                    const RougeGemmConfig *cfg) {
  if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C) return;
  const size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
  std::vector<float> Are(na), Aim(na), Bre(nb), Bim(nb);
  for (size_t i = 0; i < na; ++i) {
    Are[i] = A[i].re;
    Aim[i] = A[i].im;
  }
  for (size_t i = 0; i < nb; ++i) {
    Bre[i] = B[i].re;
    Bim[i] = B[i].im;
  }
  std::vector<float> rr(nc), ii(nc), t(nc);
  rouge_gemm_f32(M, N, K, Are.data(), Bre.data(), rr.data(), cfg);
  rouge_gemm_f32(M, N, K, Aim.data(), Bim.data(), t.data(), cfg);
  for (size_t i = 0; i < nc; ++i) rr[i] -= t[i];
  rouge_gemm_f32(M, N, K, Are.data(), Bim.data(), ii.data(), cfg);
  rouge_gemm_f32(M, N, K, Aim.data(), Bre.data(), t.data(), cfg);
  for (size_t i = 0; i < nc; ++i) {
    C[i].re = rr[i];
    C[i].im = ii[i] + t[i];
  }
}

void rouge_gemm_z64(int M, int N, int K, const RougeComplexD *A,
                    const RougeComplexD *B, RougeComplexD *C,
                    const RougeGemmConfig *cfg) {
  if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C) return;
  const size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
  std::vector<double> Are(na), Aim(na), Bre(nb), Bim(nb);
  for (size_t i = 0; i < na; ++i) {
    Are[i] = A[i].re;
    Aim[i] = A[i].im;
  }
  for (size_t i = 0; i < nb; ++i) {
    Bre[i] = B[i].re;
    Bim[i] = B[i].im;
  }
  std::vector<double> rr(nc), ii(nc), t(nc);
  rouge_gemm_f64(M, N, K, Are.data(), Bre.data(), rr.data(), cfg);
  rouge_gemm_f64(M, N, K, Aim.data(), Bim.data(), t.data(), cfg);
  for (size_t i = 0; i < nc; ++i) rr[i] -= t[i];
  rouge_gemm_f64(M, N, K, Are.data(), Bim.data(), ii.data(), cfg);
  rouge_gemm_f64(M, N, K, Aim.data(), Bre.data(), t.data(), cfg);
  for (size_t i = 0; i < nc; ++i) {
    C[i].re = rr[i];
    C[i].im = ii[i] + t[i];
  }
}

}  // extern "C"
