/*
 * cuBLAS-шим Wave 2: наш GEMM сверху, их API.
 *
 * Совместимость по публичному API cuBLAS (подход Wine: интерфейсы по открытой
 * документации, реализация своя). Покрыт минимальный честный поднабор f32:
 *   cublasCreate_v2 / cublasDestroy_v2 / cublasGetVersion_v2 /
 *   cublasSgemm_v2 / cublasGetStatusString
 * Остального здесь нет: отсутствующий символ — громкая ошибка линковки, а не
 * тихий мискомпил. Это НЕ библиотека NVIDIA.
 *
 * Семантика column-major: C = alpha*op(A)*op(B) + beta. Наш rouge_gemm_f32 —
 * row-major, поэтому считаем через транспонированную тройку:
 *   Ct = op(B)^T * op(A)^T (row-major), затем C = alpha*Ct^T + beta*C.
 * Транспозиции — явные копии, без трюков с lda.
 */

#include "rouge/cublas_shim.h"

#include <cstring>
#include <vector>

#include "rouge/numerics.h"

namespace {

struct Ctx {
  int magic;
};

// src — column-major R×C с ведущим измерением ld: элемент (i,j) в src[j*ld+i].
void transpose_copy(int R, int C, const float *src, int ld, float *dst) {
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j) dst[(size_t)j * R + i] = src[(size_t)j * ld + i];
}

}  // namespace

extern "C" {

const char *cublasGetStatusString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
  if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
  Ctx *c = new (std::nothrow) Ctx{0x43425553};
  if (!c) return CUBLAS_STATUS_ALLOC_FAILED;
  *handle = (cublasHandle_t)c;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
  if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
  delete (Ctx *)handle;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
  if (!handle || !version) return CUBLAS_STATUS_INVALID_VALUE;
  // Своя версия шима в формате cuBLAS (MAJOR*1000+MINOR*100+PATCH): 0.2.0.
  *version = 200;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc) {
  if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0 || k < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if ((transa != CUBLAS_OP_N && transa != CUBLAS_OP_T &&
       transa != CUBLAS_OP_C) ||
      (transb != CUBLAS_OP_N && transb != CUBLAS_OP_T &&
       transb != CUBLAS_OP_C))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;
  // C[f32]: для вещественных C == T.
  const int need_lda = (transa == CUBLAS_OP_N) ? m : k;
  const int need_ldb = (transb == CUBLAS_OP_N) ? k : n;
  if (lda < need_lda || ldb < need_ldb || ldc < m)
    return CUBLAS_STATUS_INVALID_VALUE;

  // op(A): m×k column-major -> построим At = op(A)^T размером k×m row-major.
  // op(B): k×n column-major -> Bt = op(B)^T размером n×k row-major.
  std::vector<float> At((size_t)k * m), Bt((size_t)n * k);
  if (transa == CUBLAS_OP_N) {
    // A column-major m×k, lda>=m: At[j][i] = A[i][j].
    transpose_copy(m, k, A, lda, At.data());
  } else {
    // A column-major k×m (транспонированная), lda>=k: op(A)[i][j]=A[j][i],
    // значит At = A как row-major k×m напрямую (копия с lda).
    for (int i = 0; i < k; ++i)
      for (int j = 0; j < m; ++j) At[(size_t)i * m + j] = A[(size_t)j * lda + i];
  }
  if (transb == CUBLAS_OP_N) {
    transpose_copy(k, n, B, ldb, Bt.data());
  } else {
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < k; ++j) Bt[(size_t)i * k + j] = B[(size_t)j * ldb + i];
  }
  // Ct (n×m) = Bt (n×k) * At (k×m), row-major нашим GEMM.
  std::vector<float> Ct((size_t)n * m);
  rouge_gemm_f32(n, m, k, Bt.data(), At.data(), Ct.data(), nullptr);
  // C = alpha*Ct^T + beta*C (всё column-major, ldc>=m).
  const float al = *alpha, be = *beta;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      C[(size_t)j * ldc + i] = al * Ct[(size_t)j * m + i] + be * C[(size_t)j * ldc + i];
  return CUBLAS_STATUS_SUCCESS;
}

}  // extern "C"
