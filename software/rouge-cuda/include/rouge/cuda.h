/*
 * rouge-cuda: CUDA Driver API — подмножество, реализуемое слоем совместимости.
 *
 * Это НЕ заголовок NVIDIA. Это наш собственный контракт API: ровно те символы,
 * которые слой перехватывает и исполняет (CPU-эмуляция или проброс на реальную libcuda).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int CUresult;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_NOT_SUPPORTED 801 /* значение из cuda.h NVIDIA */

typedef unsigned long long CUdeviceptr; /* адреса в эмуляции = host-адреса */
typedef int CUdevice; /* как в cuda.h NVIDIA: opaque ordinal */
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream; /* opaque */

/* Подмножество CUdevice_attribute из cuda.h NVIDIA: ровно те атрибуты,
 * значения которых CPU-бэкенд отвечает честно. Остальные атрибуты
 * (любые другие номера) дают CUDA_ERROR_NOT_SUPPORTED. */
typedef enum CUdevice_attribute {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
} CUdevice_attribute;

/* Подмножество CUfunction_attribute из cuda.h NVIDIA. */
typedef enum CUfunction_attribute {
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
  CU_FUNC_ATTRIBUTE_PTX_VERSION = 5
} CUfunction_attribute;

CUresult cuInit(unsigned int Flags);
CUresult cuDriverGetVersion(int *driverVersion);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuDeviceComputeCapability(int *pMajor, int *pMinor, CUdevice dev);
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, int dev);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuCtxGetDevice(CUdevice *device);
CUresult cuCtxSynchronize(void);
CUresult cuCtxDestroy(CUcontext ctx);

/* Версионированные имена, которые зовут бинарники, собранные свежими
 * заголовками NVIDIA (cuCtxCreate -> cuCtxCreate_v4, cuMemAlloc ->
 * cuMemAlloc_v2, ...): те же сигнатуры и то же поведение, что у базовых. */
CUresult cuCtxCreate_v4(CUcontext *pctx, void *params, unsigned int flags, int dev);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream);
CUresult cuMemGetInfo_v2(size_t *free, size_t *total);
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                              const char *name);
CUresult cuStreamDestroy_v2(CUstream stream);

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                           size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream);
CUresult cuMemGetInfo(size_t *free, size_t *total);
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);
CUresult cuMemFreeHost(void *p);
CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags);

CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleUnload(CUmodule module);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                           const char *name);

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult cuStreamDestroy(CUstream stream);
CUresult cuStreamSynchronize(CUstream stream);
CUresult cuStreamQuery(CUstream stream);

CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);

/* Occupancy: тот же документированный алгоритм, что в открытом cuda_occupancy.h,
 * для референсного sm_75. CUoccupancyB2DSize — как в настоящем cuda.h. */
typedef size_t (*CUoccupancyB2DSize)(int blockSize);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func,
                                                     int blockSize, size_t dynamicSMemSize);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func,
                                                               int blockSize,
                                                               size_t dynamicSMemSize,
                                                               unsigned int flags);
CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func,
                                          CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                          size_t dynamicSMemSize, int blockSizeLimit);

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, void *hStream,
                        void **kernelParams, void **extra);

CUresult cuGetErrorString(CUresult error, const char **pStr);
CUresult cuGetErrorName(CUresult error, const char **pStr);

#ifdef __cplusplus
}
#endif