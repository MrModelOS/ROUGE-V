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
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_NOT_INITIALIZED 3

typedef unsigned long long CUdeviceptr; /* адреса в эмуляции = host-адреса */
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;

CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGetCount(int *count);
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, int dev);
CUresult cuCtxSynchronize(void);
CUresult cuCtxDestroy(CUcontext ctx);

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);

CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleUnload(CUmodule module);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, void *hStream,
                        void **kernelParams, void **extra);

CUresult cuGetErrorString(CUresult error, const char **pStr);

#ifdef __cplusplus
}
#endif