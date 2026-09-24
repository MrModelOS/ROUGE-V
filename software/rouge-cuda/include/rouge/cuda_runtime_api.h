/*
 * rouge-cuda: CUDA Runtime API — подмножество, реализуемое слоем совместимости.
 * Собственный контракт API (не заголовок NVIDIA).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorCudartUnloading = 29,
  cudaErrorUnknown = 30,
  cudaErrorInvalidResourceHandle = 400
} cudaError_t;

typedef enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
} cudaMemcpyKind;

typedef void *cudaStream_t; /* opaque */

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetLastError(void);
cudaError_t cudaGetDeviceCount(int *count);
const char *cudaGetErrorString(cudaError_t err);

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);

#ifdef __cplusplus
}
#endif