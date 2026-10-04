#include <rouge/cuda_runtime_api.h>

#include <rouge/cuda.h>

#include "occupancy.h"
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
  prop->sharedMemPerMultiprocessor = 64u * 1024u;  // sm_75: 64K (замер)
  prop->regsPerBlock = 65536;
  prop->regsPerMultiprocessor = 65536;
  prop->warpSize = 32;
  prop->maxThreadsPerBlock = 1024;
  prop->maxThreadsPerMultiProcessor = 1024;  // Turing: уполовинено против Volta (замер)
  prop->maxThreadsDim[0] = 1024;
  prop->maxThreadsDim[1] = 1024;
  prop->maxThreadsDim[2] = 64;
  prop->maxGridSize[0] = 2147483647;
  prop->maxGridSize[1] = 65535;
  prop->maxGridSize[2] = 65535;
  prop->multiProcessorCount = 1;
  prop->canMapHostMemory = 1;
  prop->concurrentKernels = 1;   // real sm_75 reports 1; эмуляция последовательна, но флаг — про API
  prop->ECCEnabled = 0;
  prop->major = 7;
  prop->minor = 5;                // reference validation target (sm_75)
  // Clocks are reported as 0: no GPU clock exists to measure, and a guessed
  // number here would be a fabricated specification.
  // Clocks are left at zero from the memset: no GPU clock exists to measure,
  // and a plausible-looking number here would be a fabricated specification.
  prop->integrated = 0;          // дискретная карта (замер)
  prop->asyncEngineCount = 3;    // как у референсного sm_75 (замер)
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

// ---- function attributes + occupancy ---------------------------------------
// Ядро адресуется stub'ом (значение из __cudaRegisterFunction/__cudaGetKernel).
// Разрешение: stub -> mangled-имя -> (модуль, индекс) через общий поиск.

namespace {

bool stats_for_stub(const void* func, rouge::KernelStats* st) {
  if (func == nullptr || st == nullptr) return false;
  const auto it = rouge::stub_names().find(reinterpret_cast<uintptr_t>(func));
  if (it == rouge::stub_names().end()) return false;
  const rouge::PtxProgram* prog = nullptr;
  int fnIndex = -1;
  if (!rouge::find_kernel_program(it->second, &prog, &fnIndex)) return false;
  return rouge::kernel_stats(*prog, fnIndex, st);
}

void fill_func_attributes(rouge_cuda_func_attributes* attr, const rouge::KernelStats& st) {
  std::memset(attr, 0, sizeof(*attr));
  attr->sharedSizeBytes = st.staticSmem;
  attr->constSizeBytes = st.constSize;
  attr->localSizeBytes = st.localSize;
  attr->maxThreadsPerBlock = 1024;
  attr->numRegs = st.numRegs;
  // ptxVersion/binaryVersion — compute capability, на которую нацелен модуль
  // (.target sm_XX). Именно это возвращает настоящий драйвер для PTX-модулей:
  // проба дала ptx=75 bin=75 для -arch=sm_75 (виртуальная архитектура
  // compute_75, а не версия языка PTX 9.4).
  attr->ptxVersion = st.targetSm;
  attr->binaryVersion = st.targetSm;
  attr->cacheModeCA = 0;
  // Статика отъедает per-block лимит 48K; больше — незапускаемо, saturate в 0.
  attr->maxDynamicSharedSizeBytes =
      st.staticSmem >= 49152 ? 0 : static_cast<int>(49152 - st.staticSmem);
  attr->preferredShmemCarveout = -1;  // cudaSharedmemCarveoutDefault (проба)
  // Кластеры, барьеры Hopper+, граф-ноды: бэкенд их не поддерживает.
  // deviceNodeUpdateStatus = 0 (проба на настоящем драйвере даёт 1 — это про
  // его возможности, а не наши; честный ответ нашего слоя — 0).
  attr->sharedMemoryMode = rouge_cudaSharedMemoryModeDefault;
}

}  // namespace

cudaError_t cudaFuncGetAttributes(rouge_cuda_func_attributes *attr, const void *func) {
  if (attr == nullptr || func == nullptr) {
    tls_last_error = cudaErrorInvalidValue;
    return tls_last_error;
  }
  rouge::KernelStats st;
  if (!stats_for_stub(func, &st)) {
    tls_last_error = cudaErrorInvalidDeviceFunction;
    return tls_last_error;
  }
  fill_func_attributes(attr, st);
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func,
                                                          int blockSize,
                                                          size_t dynamicSMemSize) {
  if (numBlocks == nullptr || func == nullptr || blockSize <= 0) {
    tls_last_error = cudaErrorInvalidValue;
    return tls_last_error;
  }
  rouge::KernelStats st;
  if (!stats_for_stub(func, &st)) {
    tls_last_error = cudaErrorInvalidDeviceFunction;
    return tls_last_error;
  }
  // Невозможная конфигурация — успех с нулём, как у настоящего драйвера
  // (проба: bs=2048 -> rc=0 nb=0, dyn=1M -> rc=0 nb=0).
  *numBlocks = rouge::occ_active_blocks(st, blockSize, dynamicSMemSize);
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks,
                                                                   const void *func,
                                                                   int blockSize,
                                                                   size_t dynamicSMemSize,
                                                                   unsigned int flags) {
  // Допустимы только документированные флаги; чужой бит — InvalidValue
  // (проба: 0xdeadbeef -> rc=1). На sm_75 флаг кэширования на арифметику
  // лимитов не влияет (partitioned GC там не поддерживается).
  if (flags != cudaOccupancyDefault && flags != cudaOccupancyDisableCachingOverride) {
    tls_last_error = cudaErrorInvalidValue;
    return tls_last_error;
  }
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize,
                                                      dynamicSMemSize);
}

cudaError_t cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                               const void *func,
                                               size_t (*blockSizeToDynamicSMemSize)(int),
                                               size_t dynamicSMemSize) {
  if (minGridSize == nullptr || blockSize == nullptr || func == nullptr) {
    tls_last_error = cudaErrorInvalidValue;
    return tls_last_error;
  }
  rouge::KernelStats st;
  if (!stats_for_stub(func, &st)) {
    tls_last_error = cudaErrorInvalidDeviceFunction;
    return tls_last_error;
  }
  // minGridSize масштабируется числом SM машины; у эмулированного устройства
  // один SM (см. multiProcessorCount в props) — это и есть честный ответ
  // нашей топологии, а не 14 с настоящей карты.
  rouge::occ_potential_block_size(minGridSize, blockSize, st, blockSizeToDynamicSMemSize,
                                  dynamicSMemSize, /*numSms=*/1);
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, const void *func,
                                                      int numBlocks, int blockSize) {
  if (dynamicSmemSize == nullptr || func == nullptr) {
    tls_last_error = cudaErrorInvalidValue;
    return tls_last_error;
  }
  rouge::KernelStats st;
  if (!stats_for_stub(func, &st)) {
    tls_last_error = cudaErrorInvalidDeviceFunction;
    return tls_last_error;
  }
  // numBlocks <= 0 — отказ с нетронутым выходом (проба: rc=1).
  if (!rouge::occ_available_dynamic_smem(dynamicSmemSize, st, numBlocks, blockSize)) {
    tls_last_error = cudaErrorInvalidValue;
    return tls_last_error;
  }
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

// cudaDeviceGetAttribute: тот же честный набор, что у cuDeviceGetAttribute
// (значения enum совпадают 1:1, проверено по driver_types.h). Нужен в том
// числе настоящему шаблону cudaOccupancyMaxPotentialBlockSize, который зовёт
// его для warpSize/maxThreadsPerBlock/multiProcessorCount.
cudaError_t cudaDeviceGetAttribute(int *pi, cudaDeviceAttr attr, int device) {
  if (pi == nullptr) {
    tls_last_error = cudaErrorInvalidValue;
    return tls_last_error;
  }
  if (device != 0) {
    tls_last_error = cudaErrorInvalidDevice;
    return tls_last_error;
  }
  switch (attr) {
    case cudaDevAttrMaxThreadsPerBlock: *pi = 1024; break;
    case cudaDevAttrMaxSharedMemoryPerBlock: *pi = 49152; break;
    case cudaDevAttrWarpSize: *pi = 32; break;
    case cudaDevAttrMaxRegistersPerBlock: *pi = 65536; break;
    case cudaDevAttrMultiProcessorCount: *pi = 1; break;
    case cudaDevAttrConcurrentKernels: *pi = 1; break;
    case cudaDevAttrMaxThreadsPerMultiProcessor: *pi = 1024; break;
    case cudaDevAttrComputeCapabilityMajor: *pi = 7; break;
    case cudaDevAttrComputeCapabilityMinor: *pi = 5; break;
    default:
      tls_last_error = cudaErrorInvalidValue;
      return tls_last_error;
  }
  tls_last_error = cudaSuccess;
  return cudaSuccess;
}

}  // extern "C"