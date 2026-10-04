#include <rouge/cuda_runtime_api.h>

#include <rouge/cuda.h>

#include "rouge_internal.h"

#include <cstring>
#include <cstdio>
#include <unistd.h>
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

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  if (auto* real = real_symbol<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize"))
    return real(stream);
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

cudaError_t cudaSetDevice(int device) {
  if (device != 0) {
    tls_last_error = cudaErrorInvalidDevice;
    return tls_last_error;
  }
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
  if (device == nullptr) {
    tls_last_error = cudaErrorInvalidValue;
    return tls_last_error;
  }
  *device = 0;
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaDeviceReset(void) {
  // Everything the backend owns is process-wide host memory owned by the
  // program; a reset has nothing to invalidate. Reported honestly rather than
  // pretending to tear down a context that does not exist.
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(rouge_cuda_device_prop *prop, int device) {
  if (prop == nullptr || device != 0) {
    tls_last_error = device != 0 ? cudaErrorInvalidDevice : cudaErrorInvalidValue;
    return tls_last_error;
  }
  std::memset(prop, 0, sizeof(*prop));
  std::snprintf(prop->name, sizeof(prop->name), "ROUGE CPU-emulation device");
  prop->uuid.bytes[0] = 'R';
  prop->totalGlobalMem = static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) *
                         static_cast<size_t>(sysconf(_SC_PAGESIZE));
  prop->sharedMemPerBlock = 48u * 1024u;  // the real default for sm_70 and older
  prop->sharedMemPerBlockOptin = 64u * 1024u;
  prop->sharedMemPerMultiprocessor = 96u * 1024u;
  prop->regsPerBlock = 65536;
  prop->regsPerMultiprocessor = 65536;
  prop->warpSize = 32;
  prop->maxThreadsPerBlock = 1024;
  prop->maxThreadsPerMultiProcessor = 2048;
  prop->maxThreadsDim[0] = 1024;
  prop->maxThreadsDim[1] = 1024;
  prop->maxThreadsDim[2] = 64;
  prop->maxGridSize[0] = 2147483647;
  prop->maxGridSize[1] = 65535;
  prop->maxGridSize[2] = 65535;
  prop->multiProcessorCount = 1;
  prop->canMapHostMemory = 1;
  prop->concurrentKernels = 0;   // the backend runs one kernel at a time
  prop->ECCEnabled = 0;
  prop->major = 7;
  prop->minor = 5;                // reference validation target (sm_75)
  // Clocks are reported as 0: no GPU clock exists to measure, and a guessed
  // number here would be a fabricated specification.
  // Clocks are left at zero from the memset: no GPU clock exists to measure,
  // and a plausible-looking number here would be a fabricated specification.
  prop->integrated = 1;
  prop->asyncEngineCount = 0;
  prop->unifiedAddressing = 1;
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaPeekAtLastError(void) {
  // Peek must not clear the sticky error, which is the only reason to prefer
  // it over cudaGetLastError.
  return tls_last_error;
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
  if (ptr == nullptr) {
    tls_last_error = cudaErrorInvalidValue;
    return tls_last_error;
  }
  if (size == 0) size = 1;
  void *p = std::malloc(size);
  if (p == nullptr) {
    tls_last_error = cudaErrorMemoryAllocation;
    return tls_last_error;
  }
  *ptr = p;
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int /*flags*/) {
  return cudaMallocHost(ptr, size);
}

cudaError_t cudaFreeHost(void *ptr) {
  std::free(ptr);
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                            cudaStream_t /*stream*/) {
  // The backend executes every launch synchronously, so an async copy is a
  // plain copy: ordering with respect to kernels is already guaranteed.
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  if (devPtr == nullptr) {
    tls_last_error = cudaErrorInvalidValue;
    return tls_last_error;
  }
  std::memset(devPtr, value & 0xff, count);
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int /*flags*/) {
  return cudaStreamCreate(pStream);
}

}  // extern "C"