/*
 * cuBLAS-шим Wave 2: наш GEMM сверху, их API.
 *
 * Совместимость по публичному API cuBLAS (подход Wine: интерфейсы по открытой
 * документации, реализация своя). Покрыты честные подмножества:
 *   cublasCreate_v2 / Destroy / GetVersion / GetStatusString,
 *   S/D/C/Zgemm_v2 (все op N/T/C, alpha/beta),
 *   S/DgemmStridedBatched (пачкой независимых GEMM).
 * Остального здесь нет: отсутствующий символ — громкая ошибка линковки, а не
 * тихий мискомпил. Это НЕ библиотека NVIDIA.
 *
 * Семантика column-major: C = alpha*op(A)*op(B) + beta. Наши GEMM —
 * row-major, поэтому считаем через транспонированную тройку:
 *   Ct = op(B)^T * op(A)^T (row-major), затем C = alpha*Ct^T + beta*C.
 * Комплексные ядра идут через вещественные f64-GEMM (расширение до double,
 * сужение на выходе — точнее, чем требует f32/z64-арифметика).
 */

#include "rouge/cublas_shim.h"

#include <cstring>
#include <vector>

#include "rouge/numerics.h"

namespace {

struct Ctx {
  int magic;
};

// Вещественное ядро: pA = op(A)^T (k×m row-major), pB = op(B)^T (n×k),
// Ct = pB*pA своим GEMM, C = alpha*Ct^T + beta*C. Вызывающий проверил всё,
// кроме указателей и размеров (здесь — ещё раз дёшево).
template <typename T>
cublasStatus_t rgemm_core(cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k, T al, const T *A, int lda,
                          const T *B, int ldb, T be, T *C, int ldc,
                          void (*gemm)(int, int, int, const T *, const T *, T *,
                                       const RougeGemmConfig *)) {
  std::vector<T> At((size_t)k * m), Bt((size_t)n * k);
  if (transa == CUBLAS_OP_N) {
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < k; ++j)
        At[(size_t)j * m + i] = A[(size_t)j * lda + i];
  } else {
    for (int i = 0; i < k; ++i)
      for (int j = 0; j < m; ++j) At[(size_t)i * m + j] = A[(size_t)j * lda + i];
  }
  if (transb == CUBLAS_OP_N) {
    for (int i = 0; i < k; ++i)
      for (int j = 0; j < n; ++j)
        Bt[(size_t)j * k + i] = B[(size_t)j * ldb + i];
  } else {
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < k; ++j) Bt[(size_t)i * k + j] = B[(size_t)j * ldb + i];
  }
  std::vector<T> Ct((size_t)n * m);
  gemm(n, m, k, Bt.data(), At.data(), Ct.data(), nullptr);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      C[(size_t)j * ldc + i] =
          al * Ct[(size_t)j * m + i] + be * C[(size_t)j * ldc + i];
  return CUBLAS_STATUS_SUCCESS;
}

// Комплексное ядро в double: входы уже разложены вызывающим на Re/Im
// col-major массивы с шагом lda/ldb (рабочая область, без хвостов).
// pA = op(A)^T как row-major k×m (для op=C — с сопряжением).
cublasStatus_t cgemm_core(cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k, double al_re, double al_im,
                          const double *Ar, const double *Ai, int lda,
                          const double *Br, const double *Bi, int ldb,
                          double be_re, double be_im, double *Cr, double *Ci,
                          int ldc) {
  std::vector<double> pAr((size_t)k * m), pAi((size_t)k * m);
  std::vector<double> pBr((size_t)n * k), pBi((size_t)n * k);
  const bool ca = (transa == CUBLAS_OP_C), cb = (transb == CUBLAS_OP_C);
  for (int r = 0; r < k; ++r)
    for (int c = 0; c < m; ++c) {
      double re, im;
      if (transa == CUBLAS_OP_N) {
        re = Ar[(size_t)r * lda + c];
        im = Ai[(size_t)r * lda + c];
      } else {
        re = Ar[(size_t)c * lda + r];
        im = Ai[(size_t)c * lda + r];
        if (ca) im = -im;
      }
      pAr[(size_t)r * m + c] = re;
      pAi[(size_t)r * m + c] = im;
    }
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < k; ++c) {
      double re, im;
      if (transb == CUBLAS_OP_N) {
        re = Br[(size_t)r * ldb + c];
        im = Bi[(size_t)r * ldb + c];
      } else {
        re = Br[(size_t)c * ldb + r];
        im = Bi[(size_t)c * ldb + r];
        if (cb) im = -im;
      }
      pBr[(size_t)r * k + c] = re;
      pBi[(size_t)r * k + c] = im;
    }
  std::vector<double> ctR((size_t)n * m), ctI((size_t)n * m), t((size_t)n * m);
  rouge_gemm_f64(n, m, k, pBr.data(), pAr.data(), ctR.data(), nullptr);
  rouge_gemm_f64(n, m, k, pBi.data(), pAi.data(), t.data(), nullptr);
  for (size_t i = 0, e = ctR.size(); i < e; ++i) ctR[i] -= t[i];
  rouge_gemm_f64(n, m, k, pBr.data(), pAi.data(), ctI.data(), nullptr);
  rouge_gemm_f64(n, m, k, pBi.data(), pAr.data(), t.data(), nullptr);
  for (size_t i = 0, e = ctI.size(); i < e; ++i) ctI[i] += t[i];
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      const double pr = ctR[(size_t)j * m + i], pi = ctI[(size_t)j * m + i];
      const double cr = Cr[(size_t)j * ldc + i], ci = Ci[(size_t)j * ldc + i];
      Cr[(size_t)j * ldc + i] =
          (al_re * pr - al_im * pi) + (be_re * cr - be_im * ci);
      Ci[(size_t)j * ldc + i] =
          (al_re * pi + al_im * pr) + (be_re * ci + be_im * cr);
    }
  return CUBLAS_STATUS_SUCCESS;
}

bool valid_op(cublasOperation_t o) {
  return o == CUBLAS_OP_N || o == CUBLAS_OP_T || o == CUBLAS_OP_C;
}

cublasStatus_t check_gemm(cublasHandle_t h, cublasOperation_t ta,
                          cublasOperation_t tb, int m, int n, int k, int lda,
                          int ldb, int ldc, bool &empty) {
  empty = false;
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || k < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (!valid_op(ta) || !valid_op(tb)) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0 || k == 0) {
    empty = true;
    return CUBLAS_STATUS_SUCCESS;
  }
  if (lda < (ta == CUBLAS_OP_N ? m : k) || ldb < (tb == CUBLAS_OP_N ? k : n) ||
      ldc < m)
    return CUBLAS_STATUS_INVALID_VALUE;
  return CUBLAS_STATUS_SUCCESS;
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
  // Своя версия шима: 0.3.0 (было 0.2.0 до D/C/Z и batched).
  *version = 300;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc) {
  bool empty = false;
  const cublasStatus_t st =
      check_gemm(handle, transa, transb, m, n, k, lda, ldb, ldc, empty);
  if (st != CUBLAS_STATUS_SUCCESS || empty) return st;
  if (!alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
  return rgemm_core<float>(transa, transb, m, n, k, *alpha, A, lda, B, ldb,
                           *beta, C, ldc, rouge_gemm_f32);
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const double *alpha, const double *A, int lda,
                              const double *B, int ldb, const double *beta,
                              double *C, int ldc) {
  bool empty = false;
  const cublasStatus_t st =
      check_gemm(handle, transa, transb, m, n, k, lda, ldb, ldc, empty);
  if (st != CUBLAS_STATUS_SUCCESS || empty) return st;
  if (!alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
  return rgemm_core<double>(transa, transb, m, n, k, *alpha, A, lda, B, ldb,
                            *beta, C, ldc, rouge_gemm_f64);
}

cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *B, int ldb,
                              const cuComplex *beta, cuComplex *C, int ldc) {
  bool empty = false;
  const cublasStatus_t st =
      check_gemm(handle, transa, transb, m, n, k, lda, ldb, ldc, empty);
  if (st != CUBLAS_STATUS_SUCCESS || empty) return st;
  if (!alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
  // Рабочие области col-major без хвостов, сразу в double.
  const int a_cols = (transa == CUBLAS_OP_N) ? k : m;
  const int b_cols = (transb == CUBLAS_OP_N) ? n : k;
  std::vector<double> Ar((size_t)lda * a_cols), Ai((size_t)lda * a_cols);
  std::vector<double> Br((size_t)ldb * b_cols), Bi((size_t)ldb * b_cols);
  std::vector<double> Cr((size_t)ldc * n), Ci((size_t)ldc * n);
  for (int j = 0; j < a_cols; ++j)
    for (int i = 0; i < lda; ++i) {
      Ar[(size_t)j * lda + i] = A[(size_t)j * lda + i].x;
      Ai[(size_t)j * lda + i] = A[(size_t)j * lda + i].y;
    }
  for (int j = 0; j < b_cols; ++j)
    for (int i = 0; i < ldb; ++i) {
      Br[(size_t)j * ldb + i] = B[(size_t)j * ldb + i].x;
      Bi[(size_t)j * ldb + i] = B[(size_t)j * ldb + i].y;
    }
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < ldc; ++i) {
      Cr[(size_t)j * ldc + i] = C[(size_t)j * ldc + i].x;
      Ci[(size_t)j * ldc + i] = C[(size_t)j * ldc + i].y;
    }
  const cublasStatus_t r = cgemm_core(
      transa, transb, m, n, k, alpha->x, alpha->y, Ar.data(), Ai.data(), lda,
      Br.data(), Bi.data(), ldb, beta->x, beta->y, Cr.data(), Ci.data(), ldc);
  if (r != CUBLAS_STATUS_SUCCESS) return r;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < ldc; ++i) {
      C[(size_t)j * ldc + i].x = (float)Cr[(size_t)j * ldc + i];
      C[(size_t)j * ldc + i].y = (float)Ci[(size_t)j * ldc + i];
    }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *B, int ldb,
                              const cuDoubleComplex *beta, cuDoubleComplex *C,
                              int ldc) {
  bool empty = false;
  const cublasStatus_t st =
      check_gemm(handle, transa, transb, m, n, k, lda, ldb, ldc, empty);
  if (st != CUBLAS_STATUS_SUCCESS || empty) return st;
  if (!alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
  const int a_cols = (transa == CUBLAS_OP_N) ? k : m;
  const int b_cols = (transb == CUBLAS_OP_N) ? n : k;
  std::vector<double> Ar((size_t)lda * a_cols), Ai((size_t)lda * a_cols);
  std::vector<double> Br((size_t)ldb * b_cols), Bi((size_t)ldb * b_cols);
  std::vector<double> Cr((size_t)ldc * n), Ci((size_t)ldc * n);
  for (int j = 0; j < a_cols; ++j)
    for (int i = 0; i < lda; ++i) {
      Ar[(size_t)j * lda + i] = A[(size_t)j * lda + i].x;
      Ai[(size_t)j * lda + i] = A[(size_t)j * lda + i].y;
    }
  for (int j = 0; j < b_cols; ++j)
    for (int i = 0; i < ldb; ++i) {
      Br[(size_t)j * ldb + i] = B[(size_t)j * ldb + i].x;
      Bi[(size_t)j * ldb + i] = B[(size_t)j * ldb + i].y;
    }
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < ldc; ++i) {
      Cr[(size_t)j * ldc + i] = C[(size_t)j * ldc + i].x;
      Ci[(size_t)j * ldc + i] = C[(size_t)j * ldc + i].y;
    }
  const cublasStatus_t r = cgemm_core(
      transa, transb, m, n, k, alpha->x, alpha->y, Ar.data(), Ai.data(), lda,
      Br.data(), Bi.data(), ldb, beta->x, beta->y, Cr.data(), Ci.data(), ldc);
  if (r != CUBLAS_STATUS_SUCCESS) return r;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < ldc; ++i) {
      C[(size_t)j * ldc + i].x = Cr[(size_t)j * ldc + i];
      C[(size_t)j * ldc + i].y = Ci[(size_t)j * ldc + i];
    }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    long long strideA, const float *B, int ldb, long long strideB,
    const float *beta, float *C, int ldc, long long strideC, int batchCount) {
  bool empty = false;
  const cublasStatus_t st =
      check_gemm(handle, transa, transb, m, n, k, lda, ldb, ldc, empty);
  if (st != CUBLAS_STATUS_SUCCESS) return st;
  if (batchCount < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (batchCount == 0 || empty) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
  if (strideA < 0 || strideB < 0 || strideC < 0)
    return CUBLAS_STATUS_INVALID_VALUE;
  for (int b = 0; b < batchCount; ++b) {
    const cublasStatus_t r = cublasSgemm_v2(
        handle, transa, transb, m, n, k, alpha, A + b * strideA, lda,
        B + b * strideB, ldb, beta, C + b * strideC, ldc);
    if (r != CUBLAS_STATUS_SUCCESS) return r;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    long long strideA, const double *B, int ldb, long long strideB,
    const double *beta, double *C, int ldc, long long strideC, int batchCount) {
  bool empty = false;
  const cublasStatus_t st =
      check_gemm(handle, transa, transb, m, n, k, lda, ldb, ldc, empty);
  if (st != CUBLAS_STATUS_SUCCESS) return st;
  if (batchCount < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (batchCount == 0 || empty) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
  if (strideA < 0 || strideB < 0 || strideC < 0)
    return CUBLAS_STATUS_INVALID_VALUE;
  for (int b = 0; b < batchCount; ++b) {
    const cublasStatus_t r = cublasDgemm_v2(
        handle, transa, transb, m, n, k, alpha, A + b * strideA, lda,
        B + b * strideB, ldb, beta, C + b * strideC, ldc);
    if (r != CUBLAS_STATUS_SUCCESS) return r;
  }
  return CUBLAS_STATUS_SUCCESS;
}

}  // extern "C"
