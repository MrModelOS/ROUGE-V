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

/* --- Поэлементные операции (Wave 2, только FLOAT NCHW) --- */

typedef enum {
  CUDNN_ACTIVATION_SIGMOID = 0,
  CUDNN_ACTIVATION_RELU = 1,
  CUDNN_ACTIVATION_TANH = 2,
  CUDNN_ACTIVATION_CLIPPED_RELU = 3
} cudnnActivationMode_t;

typedef enum {
  CUDNN_NOT_PROPAGATE_NAN = 0,
  CUDNN_PROPAGATE_NAN = 1
} cudnnNanPropagation_t;

typedef struct rouge_cudnn_act *cudnnActivationDescriptor_t;

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *d);
cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t d,
                                           cudnnActivationMode_t mode,
                                           cudnnNanPropagation_t reluNanOpt,
                                           double coef);
cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t d);

/* y = alpha*act(x) + beta*y. CLIPPED_RELU режет на [-coef, coef]... точнее на
 * [0, coef] при coef>0 (как в cuDNN: min(max(0,x), coef)); coef<=0 -> RELU. */
cudnnStatus_t cudnnActivationForward(cudnnHandle_t handle,
                                     cudnnActivationDescriptor_t actDesc,
                                     const void *alpha,
                                     const cudnnTensorDescriptor_t xDesc,
                                     const void *x, const void *beta,
                                     const cudnnTensorDescriptor_t yDesc,
                                     void *y);

/* B = alpha1*A + alpha2*B; дескрипторы обязаны совпадать по dims (без
 * broadcast — иначе NOT_SUPPORTED вместо тихого неверного счёта). */
cudnnStatus_t cudnnAddTensor(cudnnHandle_t handle, const void *alpha1,
                             const cudnnTensorDescriptor_t aDesc, const void *A,
                             const void *alpha2,
                             const cudnnTensorDescriptor_t bDesc, void *B);

typedef enum {
  CUDNN_SOFTMAX_FAST = 0,
  CUDNN_SOFTMAX_ACCURATE = 1,
  CUDNN_SOFTMAX_LOG = 2
} cudnnSoftmaxAlgorithm_t;

typedef enum {
  CUDNN_SOFTMAX_MODE_INSTANCE = 0,
  CUDNN_SOFTMAX_MODE_CHANNEL = 1
} cudnnSoftmaxMode_t;

/* y = alpha*softmax(x) + beta*y. INSTANCE: по C*H*W на каждый n; CHANNEL: по C
 * на каждый (n,h,w). LOG возвращает log-softmax. */
cudnnStatus_t cudnnSoftmaxForward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);

typedef enum {
  CUDNN_BATCHNORM_SPATIAL = 0,
  CUDNN_BATCHNORM_PER_ACTIVATION = 1
} cudnnBatchNormMode_t;

/* Inference BatchNorm NCHW: y = alpha*((x-mean)/sqrt(var+eps)*scale+bias)+beta*y.
 * bnScaleBiasMeanVarDesc — 1×C×1×1 FLOAT-дескриптор четырёх векторов. */
cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *scale,
    const void *bias, const void *estimatedMean, const void *estimatedVariance,
    double epsilon);

/* --- Backward свёртки (Wave 2): обучение через свои градиенты ---
 * Перечислены только принимаемые алгоритмы + FFT (отклоняется кодом);
 * отсутствующих в заголовке алгоритмов здесь нет намеренно. */
typedef enum {
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4
} cudnnConvolutionBwdDataAlgo_t;

typedef enum {
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2
} cudnnConvolutionBwdFilterAlgo_t;

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes);
cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha, const cudnnFilterDescriptor_t wD,
    const void *w, const cudnnTensorDescriptor_t dyD, const void *dy,
    const cudnnConvolutionDescriptor_t conv, cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxD, void *dx);
cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes);
cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xD,
    const void *x, const cudnnTensorDescriptor_t dyD, const void *dy,
    const cudnnConvolutionDescriptor_t conv, cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwD, void *dw);

#ifdef __cplusplus
} /* extern "C" */
#endif
