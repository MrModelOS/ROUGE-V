/*
 * rouge-cublas: декларации шима cuBLAS (Wave 2).
 *
 * Значения перечислений — из открытой документации cuBLAS (cublas_api.h):
 * операции и статусы стабильны десятилетиями. Реализация — своя
 * (src/cublas_shim.cpp поверх rouge_gemm_f32). Покрыт только f32-поднабор;
 * остальных символов здесь нет намеренно.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15,
  CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

typedef enum {
  CUBLAS_OP_N = 0,
  CUBLAS_OP_T = 1,
  CUBLAS_OP_C = 2
} cublasOperation_t;

typedef enum {
  CUBLAS_POINTER_MODE_HOST = 0,
  CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef struct rouge_cublas_ctx *cublasHandle_t;

typedef struct { float x, y; } cuComplex;          /* layout == RougeComplex */
typedef struct { double x, y; } cuDoubleComplex;   /* layout == RougeComplexD */

const char *cublasGetStatusString(cublasStatus_t status);
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version);
/* Column-major C = alpha*op(A)*op(B) + beta. Суффиксы: S=f32, D=f64,
 * C=complex32, Z=complex64. Для C/Z транспонирование C = сопряжённое. */
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc);
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const double *alpha, const double *A, int lda,
                              const double *B, int ldb, const double *beta,
                              double *C, int ldc);
cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *B, int ldb,
                              const cuComplex *beta, cuComplex *C, int ldc);
cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *B, int ldb,
                              const cuDoubleComplex *beta, cuDoubleComplex *C,
                              int ldc);
/* Strided-batched: batchCount независимых GEMM с шагом strideA/B/C. */
cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    long long strideA, const float *B, int ldb, long long strideB,
    const float *beta, float *C, int ldc, long long strideC, int batchCount);
cublasStatus_t cublasDgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    long long strideA, const double *B, int ldb, long long strideB,
    const double *beta, double *C, int ldc, long long strideC, int batchCount);

/* --- BLAS L1/L2 (Wave 2): скаляры всегда host (pointer mode только HOST) --- */
cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle,
                                       cublasPointerMode_t *mode);
cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle,
                                       cublasPointerMode_t mode);

cublasStatus_t cublasSaxpy_v2(cublasHandle_t h, int n, const float *alpha,
                              const float *x, int incx, float *y, int incy);
cublasStatus_t cublasDaxpy_v2(cublasHandle_t h, int n, const double *alpha,
                              const double *x, int incx, double *y, int incy);
cublasStatus_t cublasCaxpy_v2(cublasHandle_t h, int n, const cuComplex *alpha,
                              const cuComplex *x, int incx, cuComplex *y,
                              int incy);
cublasStatus_t cublasZaxpy_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *x, int incx,
                              cuDoubleComplex *y, int incy);
cublasStatus_t cublasScopy_v2(cublasHandle_t h, int n, const float *x, int incx,
                              float *y, int incy);
cublasStatus_t cublasDcopy_v2(cublasHandle_t h, int n, const double *x,
                              int incx, double *y, int incy);
cublasStatus_t cublasCcopy_v2(cublasHandle_t h, int n, const cuComplex *x,
                              int incx, cuComplex *y, int incy);
cublasStatus_t cublasZcopy_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *x, int incx,
                              cuDoubleComplex *y, int incy);
cublasStatus_t cublasSswap_v2(cublasHandle_t h, int n, float *x, int incx,
                              float *y, int incy);
cublasStatus_t cublasDswap_v2(cublasHandle_t h, int n, double *x, int incx,
                              double *y, int incy);
cublasStatus_t cublasCswap_v2(cublasHandle_t h, int n, cuComplex *x, int incx,
                              cuComplex *y, int incy);
cublasStatus_t cublasZswap_v2(cublasHandle_t h, int n, cuDoubleComplex *x,
                              int incx, cuDoubleComplex *y, int incy);
cublasStatus_t cublasSscal_v2(cublasHandle_t h, int n, const float *alpha,
                              float *x, int incx);
cublasStatus_t cublasDscal_v2(cublasHandle_t h, int n, const double *alpha,
                              double *x, int incx);
cublasStatus_t cublasCscal_v2(cublasHandle_t h, int n, const cuComplex *alpha,
                              cuComplex *x, int incx);
cublasStatus_t cublasZscal_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *alpha,
                              cuDoubleComplex *x, int incx);
cublasStatus_t cublasSdot_v2(cublasHandle_t h, int n, const float *x, int incx,
                             const float *y, int incy, float *result);
cublasStatus_t cublasDdot_v2(cublasHandle_t h, int n, const double *x,
                             int incx, const double *y, int incy,
                             double *result);
cublasStatus_t cublasCdotu_v2(cublasHandle_t h, int n, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *result);
cublasStatus_t cublasCdotc_v2(cublasHandle_t h, int n, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *result);
cublasStatus_t cublasZdotu_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *result);
cublasStatus_t cublasZdotc_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *result);
cublasStatus_t cublasSnrm2_v2(cublasHandle_t h, int n, const float *x, int incx,
                              float *result);
cublasStatus_t cublasDnrm2_v2(cublasHandle_t h, int n, const double *x,
                              int incx, double *result);
cublasStatus_t cublasCnrm2_v2(cublasHandle_t h, int n, const cuComplex *x,
                              int incx, float *result);
cublasStatus_t cublasZnrm2_v2(cublasHandle_t h, int n,
                              const cuDoubleComplex *x, int incx,
                              double *result);
/* gemv column-major: m/n — dims op(A) (как в gemm); C для вещественных == T. */
cublasStatus_t cublasSgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m,
                              int n, const float *alpha, const float *A,
                              int lda, const float *x, int incx,
                              const float *beta, float *y, int incy);
cublasStatus_t cublasDgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m,
                              int n, const double *alpha, const double *A,
                              int lda, const double *x, int incx,
                              const double *beta, double *y, int incy);
cublasStatus_t cublasCgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m,
                              int n, const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *x, int incx,
                              const cuComplex *beta, cuComplex *y, int incy);
cublasStatus_t cublasZgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m,
                              int n, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *beta, cuDoubleComplex *y,
                              int incy);

#ifdef __cplusplus
} /* extern "C" */
#endif
