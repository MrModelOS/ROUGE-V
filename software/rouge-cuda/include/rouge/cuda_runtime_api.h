/*
 * rouge-cuda: CUDA Runtime API — подмножество, реализуемое слоем совместимости.
 * Собственный контракт API (не заголовок NVIDIA).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* cudaDeviceProp: раскладка публичного заголовка NVIDIA (driver_types.h),
 * скопированная поле в поле. Это ABI-контракт, а не реализация: программа,
 * собранная против настоящего cuda_runtime_api.h, пишет и читает эту структуру
 * напрямую, поэтому offsets обязаны совпадать байт в байт.
 *
 * tests/test_device_prop_layout.cpp сверяет offsetof каждого поля с настоящим
 * заголовком (тест пропускается, если CUDA-заголовков нет). */
typedef struct rouge_cuda_uuid {
  char bytes[16];
} rouge_cuda_uuid_t;

typedef struct rouge_cuda_device_prop {
  char         name[256];
  rouge_cuda_uuid_t   uuid;
  char         luid[8];
  unsigned int luidDeviceNodeMask;
  size_t       totalGlobalMem;
  size_t       sharedMemPerBlock;
  int          regsPerBlock;
  int          warpSize;
  size_t       memPitch;
  int          maxThreadsPerBlock;
  int          maxThreadsDim[3];
  int          maxGridSize[3];
  size_t       totalConstMem;
  int          major;
  int          minor;
  size_t       textureAlignment;
  size_t       texturePitchAlignment;
  int          multiProcessorCount;
  int          integrated;
  int          canMapHostMemory;
  int          maxTexture1D;
  int          maxTexture1DMipmap;
  int          maxTexture2D[2];
  int          maxTexture2DMipmap[2];
  int          maxTexture2DLinear[3];
  int          maxTexture2DGather[2];
  int          maxTexture3D[3];
  int          maxTexture3DAlt[3];
  int          maxTextureCubemap;
  int          maxTexture1DLayered[2];
  int          maxTexture2DLayered[3];
  int          maxTextureCubemapLayered[2];
  int          maxSurface1D;
  int          maxSurface2D[2];
  int          maxSurface3D[3];
  int          maxSurface1DLayered[2];
  int          maxSurface2DLayered[3];
  int          maxSurfaceCubemap;
  int          maxSurfaceCubemapLayered[2];
  size_t       surfaceAlignment;
  int          concurrentKernels;
  int          ECCEnabled;
  int          pciBusID;
  int          pciDeviceID;
  int          pciDomainID;
  int          tccDriver;
  int          asyncEngineCount;
  int          unifiedAddressing;
  int          memoryBusWidth;
  int          l2CacheSize;
  int          persistingL2CacheMaxSize;
  int          maxThreadsPerMultiProcessor;
  int          streamPrioritiesSupported;
  int          globalL1CacheSupported;
  int          localL1CacheSupported;
  size_t       sharedMemPerMultiprocessor;
  int          regsPerMultiprocessor;
  int          managedMemory;
  int          isMultiGpuBoard;
  int          multiGpuBoardGroupID;
  int          hostNativeAtomicSupported;
  int          pageableMemoryAccess;
  int          concurrentManagedAccess;
  int          computePreemptionSupported;
  int          canUseHostPointerForRegisteredMem;
  int          cooperativeLaunch;
  size_t       sharedMemPerBlockOptin;
  int          pageableMemoryAccessUsesHostPageTables;
  int          directManagedMemAccessFromHost;
  int          maxBlocksPerMultiProcessor;
  int          accessPolicyMaxWindowSize;
  size_t       reservedSharedMemPerBlock;
  int          hostRegisterSupported;
  int          sparseCudaArraySupported;
  int          hostRegisterReadOnlySupported;
  int          timelineSemaphoreInteropSupported;
  int          memoryPoolsSupported;
  int          gpuDirectRDMASupported;
  unsigned int gpuDirectRDMAFlushWritesOptions;
  int          gpuDirectRDMAWritesOrdering;
  unsigned int memoryPoolSupportedHandleTypes;
  int          deferredMappingCudaArraySupported;
  int          ipcEventSupported;
  int          clusterLaunch;
  int          unifiedFunctionPointers;
  int          deviceNumaConfig;
  int          deviceNumaId;
  int          mpsEnabled;
  int          hostNumaId;
  unsigned int gpuPciDeviceID;
  unsigned int gpuPciSubsystemID;
  int          hostNumaMultinodeIpcSupported;
  int          reserved[56];
} rouge_cuda_device_prop;

typedef enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInvalidDevice = 10,
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

/* dim3 — трехмерный размер (сетка/блок), упакован в 12 байт. Именно такой
 * тип компилятор NVIDIA кладёт в __cudaPushCallConfiguration и __cudaLaunchKernel,
 * поэтому размер и раскладка полей обязаны совпадать. */
typedef struct dim3_st {
  unsigned int x, y, z;
} dim3;

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetLastError(void);
const char *cudaGetErrorString(cudaError_t err);

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);

/* Запуск ядра по адресу-заглушке, полученному при регистрации
 * (__cudaRegisterFunction). Именно так компилятор NVIDIA реализует <<<>>>:
 * cudaLaunchKernel в объектном коде не попадает, вместо него зовут
 * __cudaPushCallConfiguration + __cudaLaunchKernel. Публичная форма нужна
 * для кода, собранного через -rdc/driver-API. */
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMemBytes, cudaStream_t stream);

/* Устройство и контекст. Наш слой эмулирует ровно одно устройство с индексом 0;
 * любой другой индекс — честный cudaErrorInvalidDevice. */
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDeviceProperties(rouge_cuda_device_prop *prop, int device);

/* Ошибки: Peek не сбрасывает состояние, Get сбрасывает. */
cudaError_t cudaPeekAtLastError(void);

/* Память хоста: pinned-память в эмуляции — обычная куча (DMA некуда гонять),
 * pinned-указатель и обычный совпадают, как и при unified memory. */
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags);
cudaError_t cudaFreeHost(void *ptr);

#define cudaHostAllocDefault 0x00
#define cudaHostAllocPortable 0x01
#define cudaHostAllocMapped 0x02
#define cudaHostAllocWriteCombined 0x04

/* Асинхронные копии: бэкенд синхронный, поэтому работают как обычные. */
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
#define cudaStreamDefault 0x00
#define cudaStreamNonBlocking 0x01

#ifdef __cplusplus
}
#endif