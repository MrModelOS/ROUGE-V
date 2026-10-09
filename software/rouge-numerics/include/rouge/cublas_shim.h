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

#ifdef __cplusplus
} /* extern "C" */
#endif
