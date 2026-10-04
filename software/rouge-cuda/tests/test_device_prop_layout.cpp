// Сверка раскладки rouge_cuda_device_prop с настоящим заголовком NVIDIA.
//
// Наш заголовок — ABI-контракт: программа, собранная против настоящего
// cuda_runtime_api.h, пишет и читает эту структуру напрямую, поэтому каждый
// offset обязан совпадать. Тест сравнивает offsetof и sizeof для всех полей
// по списку, сгенерированному из driver_types.h.
//
// Требует настоящие CUDA-заголовки при СБОРКЕ (ROUGE_CUDA_INCLUDE). Без них
// тест честно сообщает SKIP: проверять нечего, и это не провал.

#include <stddef.h>
#include <stdio.h>

#include <rouge/cuda_runtime_api.h>

// Сгенерирован на этапе configure из настоящего driver_types.h
// (см. gen_real_prop_header.py): только struct cudaDeviceProp, без
// остального настоящего заголовка, который конфликтовал бы с нашим
// в одной единице трансляции.
#include "real_cuda_device_prop.h"

#define CHECK_FIELD(field)                                                              \
  do {                                                                                  \
    if (offsetof(rouge_cuda_device_prop, field) != offsetof(cudaDeviceProp, field) ||   \
        sizeof(((rouge_cuda_device_prop*)0)->field) != sizeof(((cudaDeviceProp*)0)->field)) { \
      printf("FAIL: field " #field ": ours off=%zu size=%zu, real off=%zu size=%zu\n",   \
             offsetof(rouge_cuda_device_prop, field),                                    \
             sizeof(((rouge_cuda_device_prop*)0)->field), offsetof(cudaDeviceProp, field), \
             sizeof(((cudaDeviceProp*)0)->field));                                       \
      ++bad;                                                                             \
    }                                                                                    \
  } while (0)

int main() {
  int bad = 0;
  if (sizeof(rouge_cuda_device_prop) != sizeof(cudaDeviceProp)) {
    printf("FAIL: sizeof ours=%zu real=%zu\n", sizeof(rouge_cuda_device_prop),
           sizeof(cudaDeviceProp));
    ++bad;
  }
  CHECK_FIELD(name);
  CHECK_FIELD(uuid);
  CHECK_FIELD(luid);
  CHECK_FIELD(luidDeviceNodeMask);
  CHECK_FIELD(totalGlobalMem);
  CHECK_FIELD(sharedMemPerBlock);
  CHECK_FIELD(regsPerBlock);
  CHECK_FIELD(warpSize);
  CHECK_FIELD(memPitch);
  CHECK_FIELD(maxThreadsPerBlock);
  CHECK_FIELD(maxThreadsDim);
  CHECK_FIELD(maxGridSize);
  CHECK_FIELD(totalConstMem);
  CHECK_FIELD(major);
  CHECK_FIELD(minor);
  CHECK_FIELD(textureAlignment);
  CHECK_FIELD(texturePitchAlignment);
  CHECK_FIELD(multiProcessorCount);
  CHECK_FIELD(integrated);
  CHECK_FIELD(canMapHostMemory);
  CHECK_FIELD(maxTexture1D);
  CHECK_FIELD(maxTexture1DMipmap);
  CHECK_FIELD(maxTexture2D);
  CHECK_FIELD(maxTexture2DMipmap);
  CHECK_FIELD(maxTexture2DLinear);
  CHECK_FIELD(maxTexture2DGather);
  CHECK_FIELD(maxTexture3D);
  CHECK_FIELD(maxTexture3DAlt);
  CHECK_FIELD(maxTextureCubemap);
  CHECK_FIELD(maxTexture1DLayered);
  CHECK_FIELD(maxTexture2DLayered);
  CHECK_FIELD(maxTextureCubemapLayered);
  CHECK_FIELD(maxSurface1D);
  CHECK_FIELD(maxSurface2D);
  CHECK_FIELD(maxSurface3D);
  CHECK_FIELD(maxSurface1DLayered);
  CHECK_FIELD(maxSurface2DLayered);
  CHECK_FIELD(maxSurfaceCubemap);
  CHECK_FIELD(maxSurfaceCubemapLayered);
  CHECK_FIELD(surfaceAlignment);
  CHECK_FIELD(concurrentKernels);
  CHECK_FIELD(ECCEnabled);
  CHECK_FIELD(pciBusID);
  CHECK_FIELD(pciDeviceID);
  CHECK_FIELD(pciDomainID);
  CHECK_FIELD(tccDriver);
  CHECK_FIELD(asyncEngineCount);
  CHECK_FIELD(unifiedAddressing);
  CHECK_FIELD(memoryBusWidth);
  CHECK_FIELD(l2CacheSize);
  CHECK_FIELD(persistingL2CacheMaxSize);
  CHECK_FIELD(maxThreadsPerMultiProcessor);
  CHECK_FIELD(streamPrioritiesSupported);
  CHECK_FIELD(globalL1CacheSupported);
  CHECK_FIELD(localL1CacheSupported);
  CHECK_FIELD(sharedMemPerMultiprocessor);
  CHECK_FIELD(regsPerMultiprocessor);
  CHECK_FIELD(managedMemory);
  CHECK_FIELD(isMultiGpuBoard);
  CHECK_FIELD(multiGpuBoardGroupID);
  CHECK_FIELD(hostNativeAtomicSupported);
  CHECK_FIELD(pageableMemoryAccess);
  CHECK_FIELD(concurrentManagedAccess);
  CHECK_FIELD(computePreemptionSupported);
  CHECK_FIELD(canUseHostPointerForRegisteredMem);
  CHECK_FIELD(cooperativeLaunch);
  CHECK_FIELD(sharedMemPerBlockOptin);
  CHECK_FIELD(pageableMemoryAccessUsesHostPageTables);
  CHECK_FIELD(directManagedMemAccessFromHost);
  CHECK_FIELD(maxBlocksPerMultiProcessor);
  CHECK_FIELD(accessPolicyMaxWindowSize);
  CHECK_FIELD(reservedSharedMemPerBlock);
  CHECK_FIELD(hostRegisterSupported);
  CHECK_FIELD(sparseCudaArraySupported);
  CHECK_FIELD(hostRegisterReadOnlySupported);
  CHECK_FIELD(timelineSemaphoreInteropSupported);
  CHECK_FIELD(memoryPoolsSupported);
  CHECK_FIELD(gpuDirectRDMASupported);
  CHECK_FIELD(gpuDirectRDMAFlushWritesOptions);
  CHECK_FIELD(gpuDirectRDMAWritesOrdering);
  CHECK_FIELD(memoryPoolSupportedHandleTypes);
  CHECK_FIELD(deferredMappingCudaArraySupported);
  CHECK_FIELD(ipcEventSupported);
  CHECK_FIELD(clusterLaunch);
  CHECK_FIELD(unifiedFunctionPointers);
  CHECK_FIELD(deviceNumaConfig);
  CHECK_FIELD(deviceNumaId);
  CHECK_FIELD(mpsEnabled);
  CHECK_FIELD(hostNumaId);
  CHECK_FIELD(gpuPciDeviceID);
  CHECK_FIELD(gpuPciSubsystemID);
  CHECK_FIELD(hostNumaMultinodeIpcSupported);
  CHECK_FIELD(reserved);
  if (bad == 0)
    printf("PASS: rouge_cuda_device_prop matches the real cudaDeviceProp layout\n");
  return bad ? 1 : 0;
}
