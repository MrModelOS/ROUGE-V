/*
 * cuBLAS-шим, часть 2: BLAS L1/L2 поверх своих ядер (rouge_saxpy и др.).
 *
 * Скаляры — всегда host: cublasSetPointerMode_v2 принимает только HOST
 * (DEVICE честно отклоняется NOT_SUPPORTED — GPU-памяти у шима нет).
 * Комплексные операции идут в double-накоплении с сужением на выходе.
 * gemv — column-major, m/n описывают op(A) (единообразно с gemm):
 * перекладывается в row-major копию и считается своим rouge_sgemv/dgemv.
 */

#include "rouge/cublas_shim.h"

#include <cmath>
#include <vector>

#include "rouge/numerics.h"

namespace {

bool vec_ok(cublasHandle_t h, int n, int incx, int incy) {
  if (!h) return false;
  return n >= 0 && incx > 0 && incy > 0;
}

// Комплексное скалярное: conj_x=false -> dotu (sum x*y), true -> dotc
// (sum conj(x)*y — cuBLAS сопрягает ПЕРВЫЙ вектор). Счёт в double.
template <typename C, typename Out>
cublasStatus_t cdot_core(int n, const C *x, int incx, const C *y, int incy,
                         Out *result, bool conj_x) {
  double sr = 0.0, si = 0.0;
  for (int i = 0; i < n; ++i) {
    double xr = (double)x[(size_t)i * incx].x;
    double xi = (double)x[(size_t)i * incx].y;
    const double yr = (double)y[(size_t)i * incy].x;
    const double yi = (double)y[(size_t)i * incy].y;
    if (conj_x) xi = -xi;
    sr += xr * yr - xi * yi;
    si += xr * yi + xi * yr;
  }
  result->x = (decltype(result->x))sr;
  result->y = (decltype(result->y))si;
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle,
                                       cublasPointerMode_t *mode) {
  if (!handle || !mode) return CUBLAS_STATUS_INVALID_VALUE;
  *mode = CUBLAS_POINTER_MODE_HOST;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle,
                                       cublasPointerMode_t mode) {
  if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
  if (mode != CUBLAS_POINTER_MODE_HOST) return CUBLAS_STATUS_NOT_SUPPORTED;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy_v2(cublasHandle_t h, int n, const float *alpha,
                              const float *x, int incx, float *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  rouge_saxpy(n, *alpha, x, incx, y, incy);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t h, int n, const double *alpha,
                              const double *x, int incx, double *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  rouge_daxpy(n, *alpha, x, incx, y, incy);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCaxpy_v2(cublasHandle_t h, int n, const cuComplex *alpha,
                              const cuComplex *x, int incx, cuComplex *y,
                              int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) {
    const double xr = x[(size_t)i * incx].x, xi = x[(size_t)i * incx].y;
    double yr = y[(size_t)i * incy].x, yi = y[(size_t)i * incy].y;
    yr += alpha->x * xr - alpha->y * xi;
    yi += alpha->x * xi + alpha->y * xr;
    y[(size_t)i * incy].x = (float)yr;
    y[(size_t)i * incy].y = (float)yi;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZaxpy_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *x, int incx,
                              cuDoubleComplex *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) {
    const double xr = x[(size_t)i * incx].x, xi = x[(size_t)i * incx].y;
    y[(size_t)i * incy].x += alpha->x * xr - alpha->y * xi;
    y[(size_t)i * incy].y += alpha->x * xi + alpha->y * xr;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScopy_v2(cublasHandle_t h, int n, const float *x, int incx,
                              float *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) y[(size_t)i * incy] = x[(size_t)i * incx];
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t h, int n, const double *x,
                              int incx, double *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) y[(size_t)i * incy] = x[(size_t)i * incx];
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCcopy_v2(cublasHandle_t h, int n, const cuComplex *x,
                              int incx, cuComplex *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) y[(size_t)i * incy] = x[(size_t)i * incx];
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZcopy_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *x, int incx,
                              cuDoubleComplex *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) y[(size_t)i * incy] = x[(size_t)i * incx];
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSswap_v2(cublasHandle_t h, int n, float *x, int incx,
                              float *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) {
    const float t = x[(size_t)i * incx];
    x[(size_t)i * incx] = y[(size_t)i * incy];
    y[(size_t)i * incy] = t;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDswap_v2(cublasHandle_t h, int n, double *x, int incx,
                              double *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) {
    const double t = x[(size_t)i * incx];
    x[(size_t)i * incx] = y[(size_t)i * incy];
    y[(size_t)i * incy] = t;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCswap_v2(cublasHandle_t h, int n, cuComplex *x, int incx,
                              cuComplex *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) {
    const cuComplex t = x[(size_t)i * incx];
    x[(size_t)i * incx] = y[(size_t)i * incy];
    y[(size_t)i * incy] = t;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZswap_v2(cublasHandle_t h, int n, cuDoubleComplex *x,
                              int incx, cuDoubleComplex *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) {
    const cuDoubleComplex t = x[(size_t)i * incx];
    x[(size_t)i * incx] = y[(size_t)i * incy];
    y[(size_t)i * incy] = t;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal_v2(cublasHandle_t h, int n, const float *alpha,
                              float *x, int incx) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || incx <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !x) return CUBLAS_STATUS_INVALID_VALUE;
  rouge_sscal(n, *alpha, x, incx);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDscal_v2(cublasHandle_t h, int n, const double *alpha,
                              double *x, int incx) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || incx <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !x) return CUBLAS_STATUS_INVALID_VALUE;
  rouge_dscal(n, *alpha, x, incx);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCscal_v2(cublasHandle_t h, int n, const cuComplex *alpha,
                              cuComplex *x, int incx) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || incx <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !x) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) {
    const double xr = x[(size_t)i * incx].x, xi = x[(size_t)i * incx].y;
    x[(size_t)i * incx].x = (float)(alpha->x * xr - alpha->y * xi);
    x[(size_t)i * incx].y = (float)(alpha->x * xi + alpha->y * xr);
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZscal_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *alpha,
                              cuDoubleComplex *x, int incx) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || incx <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !x) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < n; ++i) {
    const double xr = x[(size_t)i * incx].x, xi = x[(size_t)i * incx].y;
    x[(size_t)i * incx].x = alpha->x * xr - alpha->y * xi;
    x[(size_t)i * incx].y = alpha->x * xi + alpha->y * xr;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot_v2(cublasHandle_t h, int n, const float *x, int incx,
                             const float *y, int incy, float *result) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (!x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
  *result = rouge_sdot(n, x, incx, y, incy);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot_v2(cublasHandle_t h, int n, const double *x,
                             int incx, const double *y, int incy,
                             double *result) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (!x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
  *result = rouge_ddot(n, x, incx, y, incy);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotu_v2(cublasHandle_t h, int n, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *result) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (!x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
  return cdot_core(n, x, incx, y, incy, result, false);
}

cublasStatus_t cublasCdotc_v2(cublasHandle_t h, int n, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *result) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (!x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
  return cdot_core(n, x, incx, y, incy, result, true);
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *result) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (!x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
  return cdot_core(n, x, incx, y, incy, result, false);
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *result) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!vec_ok(h, n, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
  if (!x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
  return cdot_core(n, x, incx, y, incy, result, true);
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t h, int n, const float *x, int incx,
                              float *result) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || incx <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (!x || !result) return CUBLAS_STATUS_INVALID_VALUE;
  *result = rouge_snrm2(n, x, incx);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t h, int n, const double *x,
                              int incx, double *result) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || incx <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (!x || !result) return CUBLAS_STATUS_INVALID_VALUE;
  *result = rouge_dnrm2(n, x, incx);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCnrm2_v2(cublasHandle_t h, int n, const cuComplex *x,
                              int incx, float *result) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || incx <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (!x || !result) return CUBLAS_STATUS_INVALID_VALUE;
  double scale = 0.0, ssq = 1.0;
  for (int i = 0; i < n; ++i) {
    const double xr = x[(size_t)i * incx].x, xi = x[(size_t)i * incx].y;
    const double v = std::sqrt(xr * xr + xi * xi);
    if (v != 0.0) {
      if (scale < v) {
        ssq = 1.0 + ssq * (scale / v) * (scale / v);
        scale = v;
      } else {
        ssq += (v / scale) * (v / scale);
      }
    }
  }
  *result = (float)(scale * std::sqrt(ssq));
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZnrm2_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *x, int incx,
                              double *result) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || incx <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (!x || !result) return CUBLAS_STATUS_INVALID_VALUE;
  double scale = 0.0, ssq = 1.0;
  for (int i = 0; i < n; ++i) {
    const double xr = x[(size_t)i * incx].x, xi = x[(size_t)i * incx].y;
    const double v = std::sqrt(xr * xr + xi * xi);
    if (v != 0.0) {
      if (scale < v) {
        ssq = 1.0 + ssq * (scale / v) * (scale / v);
        scale = v;
      } else {
        ssq += (v / scale) * (v / scale);
      }
    }
  }
  *result = scale * std::sqrt(ssq);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m,
                              int n, const float *alpha, const float *A,
                              int lda, const float *x, int incx,
                              const float *beta, float *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (trans != CUBLAS_OP_N && trans != CUBLAS_OP_T && trans != CUBLAS_OP_C)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !A || !x || !beta || !y) return CUBLAS_STATUS_INVALID_VALUE;
  if (incx <= 0 || incy <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  // op(A) — m×n. Хранимая A col-major: N -> m×n (lda>=m), T/C -> n×m (lda>=n).
  const int need_lda = (trans == CUBLAS_OP_N) ? m : n;
  if (lda < need_lda) return CUBLAS_STATUS_INVALID_VALUE;
  if (trans == CUBLAS_OP_N) {
    // Row-major копия = A^T (n×m); своим T-ядром: y[m] = A*x.
    std::vector<float> Bt((size_t)n * m);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        Bt[(size_t)j * m + i] = A[(size_t)j * lda + i];
    rouge_sgemv('T', n, m, *alpha, Bt.data(), m, x, incx, *beta, y, incy);
  } else {
    // Row-major копия хранимой n×m-матрицы = op(A) (m×n) напрямую:
    // op(A)[i][j] = S[j][i], S col-major n×m -> S[i*lda+j].
    std::vector<float> Ar((size_t)m * n);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        Ar[(size_t)i * n + j] = A[(size_t)i * lda + j];
    rouge_sgemv('N', m, n, *alpha, Ar.data(), n, x, incx, *beta, y, incy);
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m,
                              int n, const double *alpha, const double *A,
                              int lda, const double *x, int incx,
                              const double *beta, double *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (trans != CUBLAS_OP_N && trans != CUBLAS_OP_T && trans != CUBLAS_OP_C)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !A || !x || !beta || !y) return CUBLAS_STATUS_INVALID_VALUE;
  if (incx <= 0 || incy <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  const int need_lda = (trans == CUBLAS_OP_N) ? m : n;
  if (lda < need_lda) return CUBLAS_STATUS_INVALID_VALUE;
  if (trans == CUBLAS_OP_N) {
    std::vector<double> Bt((size_t)n * m);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        Bt[(size_t)j * m + i] = A[(size_t)j * lda + i];
    rouge_dgemv('T', n, m, *alpha, Bt.data(), m, x, incx, *beta, y, incy);
  } else {
    std::vector<double> Ar((size_t)m * n);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        Ar[(size_t)i * n + j] = A[(size_t)i * lda + j];
    rouge_dgemv('N', m, n, *alpha, Ar.data(), n, x, incx, *beta, y, incy);
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m,
                              int n, const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *x, int incx,
                              const cuComplex *beta, cuComplex *y, int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (trans != CUBLAS_OP_N && trans != CUBLAS_OP_T && trans != CUBLAS_OP_C)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !A || !x || !beta || !y) return CUBLAS_STATUS_INVALID_VALUE;
  if (incx <= 0 || incy <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  const int need_lda = (trans == CUBLAS_OP_N) ? m : n;
  if (lda < need_lda) return CUBLAS_STATUS_INVALID_VALUE;
  const bool conj = (trans == CUBLAS_OP_C);
  // op(A)[i][j] (i<m, j<n): N — A_col[j][i]; T/C — A_col[i][j] (+conj).
  for (int i = 0; i < m; ++i) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < n; ++j) {
      double ar, ai;
      if (trans == CUBLAS_OP_N) {
        ar = A[(size_t)j * lda + i].x;
        ai = A[(size_t)j * lda + i].y;
      } else {
        ar = A[(size_t)i * lda + j].x;
        ai = A[(size_t)i * lda + j].y;
        if (conj) ai = -ai;
      }
      const double xr = x[(size_t)j * incx].x, xi = x[(size_t)j * incx].y;
      sr += ar * xr - ai * xi;
      si += ar * xi + ai * xr;
    }
    double yr = y[(size_t)i * incy].x, yi = y[(size_t)i * incy].y;
    y[(size_t)i * incy].x =
        (float)((alpha->x * sr - alpha->y * si) + (beta->x * yr - beta->y * yi));
    y[(size_t)i * incy].y =
        (float)((alpha->x * si + alpha->y * sr) + (beta->x * yi + beta->y * yr));
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m,
                              int n, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *beta, cuDoubleComplex *y,
                              int incy) {
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (trans != CUBLAS_OP_N && trans != CUBLAS_OP_T && trans != CUBLAS_OP_C)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !A || !x || !beta || !y) return CUBLAS_STATUS_INVALID_VALUE;
  if (incx <= 0 || incy <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  const int need_lda = (trans == CUBLAS_OP_N) ? m : n;
  if (lda < need_lda) return CUBLAS_STATUS_INVALID_VALUE;
  const bool conj = (trans == CUBLAS_OP_C);
  for (int i = 0; i < m; ++i) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < n; ++j) {
      double ar, ai;
      if (trans == CUBLAS_OP_N) {
        ar = A[(size_t)j * lda + i].x;
        ai = A[(size_t)j * lda + i].y;
      } else {
        ar = A[(size_t)i * lda + j].x;
        ai = A[(size_t)i * lda + j].y;
        if (conj) ai = -ai;
      }
      const double xr = x[(size_t)j * incx].x, xi = x[(size_t)j * incx].y;
      sr += ar * xr - ai * xi;
      si += ar * xi + ai * xr;
    }
    double yr = y[(size_t)i * incy].x, yi = y[(size_t)i * incy].y;
    y[(size_t)i * incy].x =
        (alpha->x * sr - alpha->y * si) + (beta->x * yr - beta->y * yi);
    y[(size_t)i * incy].y =
        (alpha->x * si + alpha->y * sr) + (beta->x * yi + beta->y * yr);
  }
  return CUBLAS_STATUS_SUCCESS;
}

}  // extern "C"
