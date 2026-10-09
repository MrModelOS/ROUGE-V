/*
 * cuDNN-шим, часть 2: backward свёртки поверх своих градиентов.
 *
 * dx считается rouge_conv_bwd_data, dw — rouge_conv_bwd_filter, затем
 * y = alpha*grad + beta*y. Workspace нам не нужен — запросы размера
 * возвращают 0 (честно: рабочий буфер не требуется, а не «зарезервировано»).
 * Алгоритмы: принимаем *_ALGO_0/_ALGO_1 всегда и WINOGRAD для bwd-data при
 * 3x3 s1p1 (там тот же прямой Winograd неприменим к транспонированной, так
 * что WINOGRAD здесь — просто алиас прямого пути bwd-data; задокументировано).
 * Остальные (FFT/tiling/nonfused) — NOT_SUPPORTED.
 */

#include "rouge/cudnn_shim.h"

#include <cstddef>
#include <vector>

#include "cudnn_detail.h"
#include "rouge/numerics.h"

extern "C" {

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes) {
  if (!handle) return CUDNN_STATUS_NOT_INITIALIZED;
  if (!wDesc || !dyDesc || !convDesc || !dxDesc || !sizeInBytes)
    return CUDNN_STATUS_BAD_PARAM;
  if (algo != CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 &&
      algo != CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 &&
      algo != CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
    return CUDNN_STATUS_NOT_SUPPORTED;
  *sizeInBytes = 0;  // прямой проход без рабочего буфера
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha, const cudnnFilterDescriptor_t wD,
    const void *w, const cudnnTensorDescriptor_t dyD, const void *dy,
    const cudnnConvolutionDescriptor_t conv, cudnnConvolutionBwdDataAlgo_t algo,
    void * /*ws*/, size_t /*wsBytes*/, const void *beta,
    const cudnnTensorDescriptor_t dxD, void *dx) {
  if (!handle) return CUDNN_STATUS_NOT_INITIALIZED;
  if (!alpha || !wD || !w || !dyD || !dy || !conv || !beta || !dxD || !dx)
    return CUDNN_STATUS_BAD_PARAM;
  if (algo != CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 &&
      algo != CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 &&
      algo != CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
    return CUDNN_STATUS_NOT_SUPPORTED;
  if (wD->type != CUDNN_DATA_FLOAT || dyD->type != CUDNN_DATA_FLOAT ||
      dxD->type != CUDNN_DATA_FLOAT)
    return CUDNN_STATUS_NOT_SUPPORTED;
  if (wD->format != CUDNN_TENSOR_NCHW || dyD->format != CUDNN_TENSOR_NCHW ||
      dxD->format != CUDNN_TENSOR_NCHW)
    return CUDNN_STATUS_NOT_SUPPORTED;
  // Согласованность: dy — N×Cout×OH×OW, dx — N×Cin×H×W, W — Cout×Cin×KH×KW.
  const int N = dyD->n, Cin = wD->c, H = dxD->h, W = dxD->w, Cout = wD->k,
            KH = wD->h, KW = wD->w;
  if (dyD->c != Cout || dxD->n != N || dxD->c != Cin) return CUDNN_STATUS_BAD_PARAM;
  const int OH = (H + 2 * conv->pad_h - KH) / conv->stride_h + 1;
  const int OW = (W + 2 * conv->pad_w - KW) / conv->stride_w + 1;
  if (dyD->h != OH || dyD->w != OW) return CUDNN_STATUS_BAD_PARAM;

  const float al = *(const float *)alpha, be = *(const float *)beta;
  const size_t n = (size_t)N * Cin * H * W;
  std::vector<float> tmp(n);
  rouge_conv_bwd_data(N, Cin, H, W, Cout, KH, KW, conv->stride_h, conv->pad_h,
                      (const float *)dy, (const float *)w, tmp.data());
  float *DX = (float *)dx;
  for (size_t i = 0; i < n; ++i) DX[i] = al * tmp[i] + be * DX[i];
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes) {
  if (!handle) return CUDNN_STATUS_NOT_INITIALIZED;
  if (!xDesc || !dyDesc || !convDesc || !dwDesc || !sizeInBytes)
    return CUDNN_STATUS_BAD_PARAM;
  if (algo != CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 &&
      algo != CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
    return CUDNN_STATUS_NOT_SUPPORTED;
  *sizeInBytes = 0;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xD,
    const void *x, const cudnnTensorDescriptor_t dyD, const void *dy,
    const cudnnConvolutionDescriptor_t conv, cudnnConvolutionBwdFilterAlgo_t algo,
    void * /*ws*/, size_t /*wsBytes*/, const void *beta,
    const cudnnFilterDescriptor_t dwD, void *dw) {
  if (!handle) return CUDNN_STATUS_NOT_INITIALIZED;
  if (!alpha || !xD || !x || !dyD || !dy || !conv || !beta || !dwD || !dw)
    return CUDNN_STATUS_BAD_PARAM;
  if (algo != CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 &&
      algo != CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
    return CUDNN_STATUS_NOT_SUPPORTED;
  if (xD->type != CUDNN_DATA_FLOAT || dyD->type != CUDNN_DATA_FLOAT ||
      dwD->type != CUDNN_DATA_FLOAT)
    return CUDNN_STATUS_NOT_SUPPORTED;
  const int N = xD->n, Cin = xD->c, H = xD->h, W = xD->w, Cout = dwD->k,
            KH = dwD->h, KW = dwD->w;
  if (dwD->c != Cin || dyD->n != N || dyD->c != Cout)
    return CUDNN_STATUS_BAD_PARAM;
  const int OH = (H + 2 * conv->pad_h - KH) / conv->stride_h + 1;
  const int OW = (W + 2 * conv->pad_w - KW) / conv->stride_w + 1;
  if (dyD->h != OH || dyD->w != OW) return CUDNN_STATUS_BAD_PARAM;

  const float al = *(const float *)alpha, be = *(const float *)beta;
  const size_t n = (size_t)Cout * Cin * KH * KW;
  std::vector<float> tmp(n);
  rouge_conv_bwd_filter(N, Cin, H, W, Cout, KH, KW, conv->stride_h, conv->pad_h,
                        (const float *)x, (const float *)dy, tmp.data());
  float *DW = (float *)dw;
  for (size_t i = 0; i < n; ++i) DW[i] = al * tmp[i] + be * DW[i];
  return CUDNN_STATUS_SUCCESS;
}

}  // extern "C"
