/*
 * cuDNN-шим Wave 2: наша свёртка сверху, их API (минимальный f32 NCHW).
 *
 * Подход тот же, что у cublas-шима: дескрипторы по открытой документации,
 * счёт — своим rouge_conv2d_f32 / winograd. y = alpha*conv(x,w) + beta*y.
 * Поддержаны только FLOAT+NCHW, dilation 1, mode 0 (cross-correlation);
 * всё остальное — CUDNN_STATUS_NOT_SUPPORTED вместо тихого вранья.
 */

#include "rouge/cudnn_shim.h"

#include <cstring>
#include <vector>

#include "rouge/numerics.h"

struct rouge_cudnn_ctx {
  int magic;
};
struct rouge_cudnn_tensor {
  int n, c, h, w;
  cudnnDataType_t type;
  cudnnTensorFormat_t format;
};
struct rouge_cudnn_filter {
  int k, c, h, w;
  cudnnDataType_t type;
  cudnnTensorFormat_t format;
};
struct rouge_cudnn_conv {
  int pad_h, pad_w, stride_h, stride_w;
};

extern "C" {

const char *cudnnGetErrorString(cudnnStatus_t s) {
  switch (s) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
    default:
      return "CUDNN_STATUS_UNKNOWN";
  }
}

cudnnStatus_t cudnnCreate(cudnnHandle_t *h) {
  if (!h) return CUDNN_STATUS_BAD_PARAM;
  auto *c = new (std::nothrow) rouge_cudnn_ctx{1};
  if (!c) return CUDNN_STATUS_ALLOC_FAILED;
  *h = c;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t h) {
  if (!h) return CUDNN_STATUS_BAD_PARAM;
  delete h;
  return CUDNN_STATUS_SUCCESS;
}

size_t cudnnGetVersion(void) { return 200; }  // своя версия шима 0.2.0

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  auto *t = new (std::nothrow) rouge_cudnn_tensor{0, 0, 0, 0, CUDNN_DATA_FLOAT,
                                                 CUDNN_TENSOR_NCHW};
  if (!t) return CUDNN_STATUS_ALLOC_FAILED;
  *d = t;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
                                         cudnnTensorFormat_t f,
                                         cudnnDataType_t t, int n, int c,
                                         int h, int w) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  if (n <= 0 || c <= 0 || h <= 0 || w <= 0) return CUDNN_STATUS_BAD_PARAM;
  d->format = f;
  d->type = t;
  d->n = n;
  d->c = c;
  d->h = h;
  d->w = w;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  delete d;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  auto *f = new (std::nothrow) rouge_cudnn_filter{0, 0, 0, 0, CUDNN_DATA_FLOAT,
                                                 CUDNN_TENSOR_NCHW};
  if (!f) return CUDNN_STATUS_ALLOC_FAILED;
  *d = f;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t d,
                                         cudnnDataType_t t,
                                         cudnnTensorFormat_t f, int k, int c,
                                         int h, int w) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  if (k <= 0 || c <= 0 || h <= 0 || w <= 0) return CUDNN_STATUS_BAD_PARAM;
  d->type = t;
  d->format = f;
  d->k = k;
  d->c = c;
  d->h = h;
  d->w = w;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  delete d;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  auto *c = new (std::nothrow) rouge_cudnn_conv{0, 0, 1, 1};
  if (!c) return CUDNN_STATUS_ALLOC_FAILED;
  *d = c;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t d,
                                              int pad_h, int pad_w,
                                              int stride_h, int stride_w,
                                              int dil_h, int dil_w, int mode,
                                              cudnnDataType_t) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  if (pad_h < 0 || pad_w < 0 || stride_h <= 0 || stride_w <= 0)
    return CUDNN_STATUS_BAD_PARAM;
  if (pad_h != pad_w || stride_h != stride_w) return CUDNN_STATUS_NOT_SUPPORTED;
  if (dil_h != 1 || dil_w != 1) return CUDNN_STATUS_NOT_SUPPORTED;
  if (mode != 0) return CUDNN_STATUS_NOT_SUPPORTED;  // только cross-correlation
  d->pad_h = pad_h;
  d->pad_w = pad_w;
  d->stride_h = stride_h;
  d->stride_w = stride_w;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  delete d;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t conv, const cudnnTensorDescriptor_t in,
    const cudnnFilterDescriptor_t f, int *n, int *c, int *h, int *w) {
  if (!conv || !in || !f || !n || !c || !h || !w)
    return CUDNN_STATUS_BAD_PARAM;
  *n = in->n;
  *c = f->k;
  *h = (in->h + 2 * conv->pad_h - f->h) / conv->stride_h + 1;
  *w = (in->w + 2 * conv->pad_w - f->w) / conv->stride_w + 1;
  if (*h <= 0 || *w <= 0) return CUDNN_STATUS_BAD_PARAM;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xD,
    const void *x, const cudnnFilterDescriptor_t wD, const void *w,
    const cudnnConvolutionDescriptor_t conv, cudnnConvolutionFwdAlgo_t algo,
    void * /*ws*/, size_t /*wsBytes*/, const void *beta,
    const cudnnTensorDescriptor_t yD, void *y) {
  if (!handle) return CUDNN_STATUS_NOT_INITIALIZED;
  if (!alpha || !xD || !x || !wD || !w || !conv || !beta || !yD || !y)
    return CUDNN_STATUS_BAD_PARAM;
  if (xD->type != CUDNN_DATA_FLOAT || wD->type != CUDNN_DATA_FLOAT ||
      yD->type != CUDNN_DATA_FLOAT)
    return CUDNN_STATUS_NOT_SUPPORTED;
  if (xD->format != CUDNN_TENSOR_NCHW || yD->format != CUDNN_TENSOR_NCHW)
    return CUDNN_STATUS_NOT_SUPPORTED;
  if (xD->c != wD->c || yD->c != wD->k || yD->n != xD->n)
    return CUDNN_STATUS_BAD_PARAM;
  if (algo != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM &&
      algo != CUDNN_CONVOLUTION_FWD_ALGO_GEMM &&
      algo != CUDNN_CONVOLUTION_FWD_ALGO_DIRECT &&
      algo != CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
    return CUDNN_STATUS_NOT_SUPPORTED;

  const float al = *(const float *)alpha, be = *(const float *)beta;
  const int N = xD->n, Cin = xD->c, H = xD->h, W = xD->w, Cout = wD->k,
            KH = wD->h, KW = wD->w;
  const int OH = (H + 2 * conv->pad_h - KH) / conv->stride_h + 1;
  const int OW = (W + 2 * conv->pad_w - KW) / conv->stride_w + 1;
  if (yD->h != OH || yD->w != OW) return CUDNN_STATUS_BAD_PARAM;

  std::vector<float> tmp((size_t)N * Cout * OH * OW);
  bool use_wino = (algo == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD) && KH == 3 &&
                  KW == 3 && conv->stride_h == 1 && conv->pad_h == 1;
  if (use_wino) {
    if (rouge_conv_winograd3x3_f32(N, Cin, H, W, Cout, (const float *)x,
                                   (const float *)w, nullptr, tmp.data()) != 0)
      return CUDNN_STATUS_EXECUTION_FAILED;
  } else {
    rouge_conv2d_f32(N, Cin, H, W, Cout, KH, KW, conv->stride_h, conv->pad_h,
                     (const float *)x, (const float *)w, nullptr, tmp.data(),
                     nullptr);
  }
  float *Y = (float *)y;
  for (size_t i = 0, n = tmp.size(); i < n; ++i) Y[i] = al * tmp[i] + be * Y[i];
  return CUDNN_STATUS_SUCCESS;
}

}  // extern "C"
