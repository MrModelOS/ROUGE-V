/*
 * rouge-cudnn: декларации минимального шима cuDNN (Wave 2).
 *
 * Значения — из открытой документации cuDNN: типы данных, форматы, алгоритмы
 * прямого прохода стабильны. Покрыт только f32 NCHW conv-forward; остального
 * здесь нет намеренно (громкая ошибка линковки вместо тихого мискомпила).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUDNN_STATUS_SUCCESS = 0,
  CUDNN_STATUS_NOT_INITIALIZED = 1,
  CUDNN_STATUS_ALLOC_FAILED = 2,
  CUDNN_STATUS_BAD_PARAM = 3,
  CUDNN_STATUS_INTERNAL_ERROR = 4,
  CUDNN_STATUS_INVALID_VALUE = 5,
  CUDNN_STATUS_ARCH_MISMATCH = 6,
  CUDNN_STATUS_MAPPING_ERROR = 7,
  CUDNN_STATUS_EXECUTION_FAILED = 8,
  CUDNN_STATUS_NOT_SUPPORTED = 9,
  CUDNN_STATUS_LICENSE_ERROR = 10
} cudnnStatus_t;

typedef enum {
  CUDNN_DATA_FLOAT = 0,
  CUDNN_DATA_DOUBLE = 1,
  CUDNN_DATA_HALF = 2,
  CUDNN_DATA_INT8 = 3
} cudnnDataType_t;

typedef enum { CUDNN_TENSOR_NCHW = 0, CUDNN_TENSOR_NHWC = 1 } cudnnTensorFormat_t;

typedef enum {
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
  CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
  CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6
} cudnnConvolutionFwdAlgo_t;

typedef struct rouge_cudnn_ctx *cudnnHandle_t;
typedef struct rouge_cudnn_tensor *cudnnTensorDescriptor_t;
typedef struct rouge_cudnn_filter *cudnnFilterDescriptor_t;
typedef struct rouge_cudnn_conv *cudnnConvolutionDescriptor_t;

const char *cudnnGetErrorString(cudnnStatus_t status);
cudnnStatus_t cudnnCreate(cudnnHandle_t *handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
size_t cudnnGetVersion(void);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *desc);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t desc,
                                         cudnnTensorFormat_t format,
                                         cudnnDataType_t type, int n, int c,
                                         int h, int w);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t desc);

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *desc);
cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t desc,
                                         cudnnDataType_t type,
                                         cudnnTensorFormat_t format, int k,
                                         int c, int h, int w);
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t desc);

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *d);
cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t d,
                                              int pad_h, int pad_w, int stride_h,
                                              int stride_w, int dilation_h,
                                              int dilation_w, int mode,
                                              cudnnDataType_t type);
cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t d);

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w);

/* Только f32 NCHW, dilation==1, mode==0 (cross-correlation). */
cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);

#ifdef __cplusplus
} /* extern "C" */
#endif
