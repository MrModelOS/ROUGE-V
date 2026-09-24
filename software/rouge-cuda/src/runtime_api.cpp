#include <rouge/cuda_runtime_api.h>

#include <rouge/cuda.h>

#include "rouge_internal.h"

#include <cstring>
#include <string>

namespace {

thread_local cudaError_t tls_last_error = cudaSuccess;

template <typename T>
T real_symbol(const char* name) {
  return reinterpret_cast<T>(rouge::load_real_symbol(name));
}

cudaError_t map_driver(CUresult r) {
  switch (r) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUDA_ERROR_INVALID_HANDLE: return cudaErrorInvalidResourceHandle;
    default: return cudaErrorUnknown;
  }
}

}  // namespace

extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size) {
  if (auto* real = real_symbol<decltype(&cudaMalloc)>("cudaMalloc")) return real(devPtr, size);
  if (!devPtr) { tls_last_error = cudaErrorInvalidValue; return tls_last_error; }
  CUdeviceptr p = 0;
  const CUresult r = cuMemAlloc(&p, size);
  tls_last_error = map_driver(r);
  if (r == CUDA_SUCCESS) *devPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(p));
  return tls_last_error;
}

cudaError_t cudaFree(void* devPtr) {
  if (auto* real = real_symbol<decltype(&cudaFree)>("cudaFree")) return real(devPtr);
  if (!devPtr) { tls_last_error = cudaErrorInvalidValue; return tls_last_error; }
  tls_last_error = map_driver(cuMemFree(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(devPtr))));
  return tls_last_error;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind /*kind*/) {
  if (auto* real = real_symbol<decltype(&cudaMemcpy)>("cudaMemcpy"))
    return real(dst, src, count, cudaMemcpyDeviceToDevice);
  if (count && (!dst || !src)) { tls_last_error = cudaErrorInvalidValue; return tls_last_error; }
  // All device pointers are host addresses in CPU emulation: one memcpy covers all kinds.
  std::memcpy(dst, src, count);
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
  if (auto* real = real_symbol<decltype(&cudaDeviceSynchronize)>("cudaDeviceSynchronize"))
    return real();
  tls_last_error = cudaSuccess;  // synchronous emulation
  return cudaSuccess;
}

cudaError_t cudaGetLastError(void) {
  if (auto* real = real_symbol<decltype(&cudaGetLastError)>("cudaGetLastError")) return real();
  const cudaError_t e = tls_last_error;
  tls_last_error = cudaSuccess;
  return e;
}

cudaError_t cudaGetDeviceCount(int* count) {
  if (auto* real = real_symbol<decltype(&cudaGetDeviceCount)>("cudaGetDeviceCount"))
    return real(count);
  if (!count) { tls_last_error = cudaErrorInvalidValue; return tls_last_error; }
  tls_last_error = map_driver(cuDeviceGetCount(count));
  return tls_last_error;
}

const char* cudaGetErrorString(cudaError_t err) {
  if (auto* real = real_symbol<decltype(&cudaGetErrorString)>("cudaGetErrorString"))
    return real(err);
  switch (err) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorInvalidDevicePointer: return "invalid device pointer";
    case cudaErrorInvalidResourceHandle: return "invalid resource handle";
    case cudaErrorUnknown: return "unknown error";
    default: return "unrecognized error code";
  }
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
  if (auto* real = real_symbol<decltype(&cudaStreamCreate)>("cudaStreamCreate"))
    return real(pStream);
  if (!pStream) { tls_last_error = cudaErrorInvalidValue; return tls_last_error; }
  *pStream = new int(0);  // opaque token; everything is synchronous
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t /*stream*/) {
  if (auto* real = real_symbol<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize"))
    return real(nullptr);
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  if (auto* real = real_symbol<decltype(&cudaStreamDestroy)>("cudaStreamDestroy"))
    return real(stream);
  delete static_cast<int*>(stream);
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

}  // extern "C"