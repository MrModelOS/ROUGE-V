/* Внутренние структуры дескрипторов cuDNN-шима. Только для .cpp файлов
 * библиотеки rouge-cudnn, не для публичного заголовка. */
#pragma once

#include "rouge/cudnn_shim.h"

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
struct rouge_cudnn_act {
  cudnnActivationMode_t mode;
  double coef;
};
