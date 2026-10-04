#include <rouge/cuda.h>

#include "occupancy.h"
#include "ptx.h"
#include "rouge_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace rouge;

namespace {

void banner() {
  static bool shown = false;
  if (!shown) {
    shown = true;
    std::fprintf(stderr, "[ROUGE-CUDA compat layer] CPU-emulation backend active\n");
  }
}

// Текущий контекст потока: единственный dummy-контекст слоя.
// Реальный драйвер хранит контекст per-thread, мы — один глобальный,
// поэтому thread_local здесь честен: виден только наш контекст.
CUcontext& tls_current() {
  thread_local CUcontext cur = nullptr;
  return cur;
}

template <typename T>
T real_symbol(const char* name) {
  return reinterpret_cast<T>(load_real_symbol(name));
}

}  // namespace

extern "C" {

CUresult cuInit(unsigned int /*Flags*/) {
  if (auto* real = real_symbol<decltype(&cuInit)>("cuInit")) return real(0);
  banner();
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int* count) {
  if (auto* real = real_symbol<decltype(&cuDeviceGetCount)>("cuDeviceGetCount")) return real(count);
  if (!count) return CUDA_ERROR_INVALID_VALUE;
  *count = 1;  // one CPU-emulated device
  return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  if (auto* real = real_symbol<decltype(&cuDeviceGet)>("cuDeviceGet")) return real(device, ordinal);
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  if (ordinal != 0) return CUDA_ERROR_INVALID_VALUE;  // a single emulated device
  *device = 0;  // device handles are opaque ordinals here
  return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, int dev) {
  if (auto* real = real_symbol<decltype(&cuCtxCreate)>("cuCtxCreate"))
    return real(pctx, flags, dev);
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = reinterpret_cast<CUcontext>(&ctx());  // dummy but stable
  tls_current() = *pctx;  // реальный драйвер делает новый контекст текущим
  return CUDA_SUCCESS;
}

// CUDA 13 header maps cuCtxCreate -> cuCtxCreate_v4
// (CUcontext*, CUctxCreateParams*, unsigned, CUdevice): binaries built
// against real headers link this name. Params are accepted and ignored;
// the CPU backend has no per-context affinity to configure.
CUresult cuCtxCreate_v4(CUcontext* pctx, void* params, unsigned int flags, int dev) {
  if (auto* real = real_symbol<CUresult (*)(CUcontext*, void*, unsigned int, int)>("cuCtxCreate_v4"))
    return real(pctx, params, flags, dev);
  return cuCtxCreate(pctx, flags, dev);
}

// The _v2 memcpy/alloc spellings binaries bind to: identical signatures,
// served by the same implementation (and the same forwarding path).
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
  if (auto* real = real_symbol<decltype(&cuMemAlloc_v2)>("cuMemAlloc_v2"))
    return real(dptr, bytesize);
  return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  if (auto* real = real_symbol<decltype(&cuMemcpyHtoD_v2)>("cuMemcpyHtoD_v2"))
    return real(dstDevice, srcHost, ByteCount);
  return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  if (auto* real = real_symbol<decltype(&cuMemcpyDtoH_v2)>("cuMemcpyDtoH_v2"))
    return real(dstHost, srcDevice, ByteCount);
  return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult cuCtxSynchronize(void) {
  if (auto* real = real_symbol<decltype(&cuCtxSynchronize)>("cuCtxSynchronize")) return real();
  return CUDA_SUCCESS;  // everything is synchronous in CPU emulation
}

CUresult cuCtxDestroy(CUcontext ctx) {
  if (auto* real = real_symbol<decltype(&cuCtxDestroy)>("cuCtxDestroy")) return real(ctx);
  if (tls_current() == ctx) tls_current() = nullptr;
  return CUDA_SUCCESS;
}

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
  if (auto* real = real_symbol<decltype(&cuMemAlloc)>("cuMemAlloc")) return real(dptr, bytesize);
  if (!dptr) return CUDA_ERROR_INVALID_VALUE;
  if (bytesize == 0) bytesize = 1;
  void* p = std::malloc(bytesize);
  if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
  const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  ctx().allocations.insert(addr);
  *dptr = static_cast<CUdeviceptr>(addr);
  return CUDA_SUCCESS;
}

CUresult cuMemFree(CUdeviceptr dptr) {
  if (auto* real = real_symbol<decltype(&cuMemFree)>("cuMemFree")) return real(dptr);
  const uintptr_t addr = static_cast<uintptr_t>(dptr);
  if (ctx().allocations.erase(addr) == 0) return CUDA_ERROR_INVALID_VALUE;
  std::free(reinterpret_cast<void*>(addr));
  return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  if (auto* real = real_symbol<decltype(&cuMemcpyHtoD)>("cuMemcpyHtoD"))
    return real(dstDevice, srcHost, ByteCount);
  if (!srcHost || !dstDevice) return CUDA_ERROR_INVALID_VALUE;
  std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice)), srcHost, ByteCount);
  return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  if (auto* real = real_symbol<decltype(&cuMemcpyDtoH)>("cuMemcpyDtoH"))
    return real(dstHost, srcDevice, ByteCount);
  if (!dstHost || !srcDevice) return CUDA_ERROR_INVALID_VALUE;
  std::memcpy(dstHost, reinterpret_cast<void*>(static_cast<uintptr_t>(srcDevice)), ByteCount);
  return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
  if (auto* real = real_symbol<decltype(&cuModuleLoadData)>("cuModuleLoadData"))
    return real(module, image);
  if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
  std::string err;
  auto prog = parse_ptx(static_cast<const char*>(image), &err);
  if (!prog) {
    std::fprintf(stderr, "[ROUGE-CUDA] PTX load failed: %s\n", err.c_str());
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto shared = std::shared_ptr<PtxProgram>(std::move(prog));
  const uint64_t key = reinterpret_cast<uint64_t>(shared.get());
  ctx().modules[key] = shared;
  *module = reinterpret_cast<CUmodule>(shared.get());
  return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule module) {
  if (auto* real = real_symbol<decltype(&cuModuleUnload)>("cuModuleUnload")) return real(module);
  const uint64_t key = reinterpret_cast<uint64_t>(module);
  const auto mit = ctx().modules.find(key);
  if (mit == ctx().modules.end()) return CUDA_ERROR_INVALID_VALUE;
  // Drop this module's global backing storage first (host pointers into it
  // dangle after unload — same as real CUDA semantics).
  const auto& prog = mit->second;
  for (size_t i = 0; i < prog->globals.size(); ++i)
    ctx().globalStorage.erase(global_storage_key(key, static_cast<int>(i)));
  ctx().modules.erase(mit);
  return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
  if (auto* real = real_symbol<decltype(&cuModuleGetFunction)>("cuModuleGetFunction"))
    return real(hfunc, hmod, name);
  if (!hfunc || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
  const auto mit = ctx().modules.find(reinterpret_cast<uint64_t>(hmod));
  if (mit == ctx().modules.end()) return CUDA_ERROR_INVALID_HANDLE;
  const auto fit = mit->second->functionIndex.find(name);
  if (fit == mit->second->functionIndex.end()) return CUDA_ERROR_INVALID_VALUE;

  auto* info = new Context::FuncInfo{reinterpret_cast<uint64_t>(hmod), fit->second};
  ctx().functions[reinterpret_cast<uint64_t>(info)] = *info;
  *hfunc = reinterpret_cast<CUfunction>(info);
  return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, void* hStream,
                        void** kernelParams, void** extra) {
  if (auto* real = real_symbol<decltype(&cuLaunchKernel)>("cuLaunchKernel"))
    return real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream, kernelParams, extra);

  const auto fit = ctx().functions.find(reinterpret_cast<uint64_t>(f));
  if (fit == ctx().functions.end()) return CUDA_ERROR_INVALID_HANDLE;
  const auto mit = ctx().modules.find(fit->second.moduleKey);
  if (mit == ctx().modules.end()) return CUDA_ERROR_INVALID_HANDLE;
  auto& prog = mit->second;
  const int fnIndex = fit->second.fnIndex;
  if (fnIndex < 0 || fnIndex >= static_cast<int>(prog->functions.size()))
    return CUDA_ERROR_INVALID_VALUE;
  const auto& fn = prog->functions[fnIndex];

  std::vector<uint8_t> blob(static_cast<size_t>(fn.paramsBlobSize), 0);
  if (kernelParams) {
    for (const auto& p : fn.params) {
      if (p.offset + p.width > static_cast<int>(blob.size()) || !kernelParams[p.index])
        return CUDA_ERROR_INVALID_VALUE;
      std::memcpy(blob.data() + p.offset, kernelParams[p.index], static_cast<size_t>(p.width));
    }
  }

  std::string err;
  // Module-global backing storage for this program (allocated on demand by
  // cuModuleGetGlobal; kernels without globals launch with an empty map,
  // which preserves the old resolve-to-zero behaviour exactly).
  std::unordered_map<std::string, uint64_t> bases;
  for (size_t i = 0; i < prog->globals.size(); ++i) {
    const uint64_t key = global_storage_key(fit->second.moduleKey, static_cast<int>(i));
    const auto git = ctx().globalStorage.find(key);
    if (git != ctx().globalStorage.end() && !git->second.empty())
      bases[prog->globals[i].name] =
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(git->second.data()));
  }
  if (!execute_kernel(*prog, fnIndex, blob, gridDimX, gridDimY, gridDimZ,
                      blockDimX, blockDimY, blockDimZ, &err, &bases)) {
    std::fprintf(stderr, "[ROUGE-CUDA] kernel launch failed: %s\n", err.c_str());
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

// ---- device / context queries -------------------------------------------
// In emulation there is no GPU behind the handle; every answer below is the
// documented contract of the CPU backend, not a measurement of hardware.

CUresult cuDriverGetVersion(int* driverVersion) {
  if (auto* real = real_symbol<decltype(&cuDriverGetVersion)>("cuDriverGetVersion"))
    return real(driverVersion);
  if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
  // The loader accepts PTX up to .version 9.4 (nvcc 13.4 output), so the
  // emulated driver reports a CUDA 13.4-level API version (13040).
  *driverVersion = 13040;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
  if (auto* real = real_symbol<decltype(&cuDeviceGetName)>("cuDeviceGetName"))
    return real(name, len, dev);
  if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  if (dev != 0) return CUDA_ERROR_INVALID_VALUE;
  const char* emu = "ROUGE CPU-emulation device";
  std::strncpy(name, emu, static_cast<size_t>(len));
  name[len - 1] = '\0';
  return CUDA_SUCCESS;
}

CUresult cuDeviceComputeCapability(int* pMajor, int* pMinor, CUdevice dev) {
  if (auto* real = real_symbol<decltype(&cuDeviceComputeCapability)>("cuDeviceComputeCapability"))
    return real(pMajor, pMinor, dev);
  if (!pMajor || !pMinor) return CUDA_ERROR_INVALID_VALUE;
  if (dev != 0) return CUDA_ERROR_INVALID_VALUE;
  // Reference validation target of this project (GeForce MX450, sm_75):
  // the highest PTX feature level the CPU backend is tested against.
  *pMajor = 7;
  *pMinor = 5;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
  if (auto* real = real_symbol<decltype(&cuDeviceGetAttribute)>("cuDeviceGetAttribute"))
    return real(pi, attrib, dev);
  if (!pi) return CUDA_ERROR_INVALID_VALUE;
  if (dev != 0) return CUDA_ERROR_INVALID_VALUE;
  // Only the attributes declared in our cuda.h are answered; anything else
  // is a loud refusal, never a guessed number.
  switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = 1024; return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE: *pi = 32; return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: *pi = 1; return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: *pi = 0; return CUDA_SUCCESS;  // synchronous
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: *pi = 2048; return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *pi = 7; return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *pi = 5; return CUDA_SUCCESS;
    default: return CUDA_ERROR_NOT_SUPPORTED;
  }
}

// Host RAM backs every emulated allocation, so the honest "device memory"
// is the physical memory of the machine (queried, not guessed).
size_t host_phys_mem() {
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page <= 0) return size_t(2) << 30;  // 2 GiB fallback
  return static_cast<size_t>(pages) * static_cast<size_t>(page);
}

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
  if (auto* real = real_symbol<decltype(&cuDeviceTotalMem)>("cuDeviceTotalMem"))
    return real(bytes, dev);
  if (!bytes) return CUDA_ERROR_INVALID_VALUE;
  if (dev != 0) return CUDA_ERROR_INVALID_VALUE;
  *bytes = host_phys_mem();
  return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
  if (auto* real = real_symbol<decltype(&cuDeviceTotalMem_v2)>("cuDeviceTotalMem_v2"))
    return real(bytes, dev);
  return cuDeviceTotalMem(bytes, dev);
}

CUresult cuCtxSetCurrent(CUcontext pctx) {
  if (auto* real = real_symbol<decltype(&cuCtxSetCurrent)>("cuCtxSetCurrent"))
    return real(pctx);
  tls_current() = pctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
  if (auto* real = real_symbol<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent"))
    return real(pctx);
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = tls_current();
  return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice* device) {
  if (auto* real = real_symbol<decltype(&cuCtxGetDevice)>("cuCtxGetDevice"))
    return real(device);
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  *device = 0;  // the single emulated device
  return CUDA_SUCCESS;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
  if (auto* real = real_symbol<decltype(&cuCtxDestroy_v2)>("cuCtxDestroy_v2"))
    return real(ctx);
  return cuCtxDestroy(ctx);
}

// ---- memory ---------------------------------------------------------------

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  if (auto* real = real_symbol<decltype(&cuMemcpyDtoD)>("cuMemcpyDtoD"))
    return real(dstDevice, srcDevice, ByteCount);
  if (!dstDevice || !srcDevice) return CUDA_ERROR_INVALID_VALUE;
  std::memmove(reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice)),
               reinterpret_cast<void*>(static_cast<uintptr_t>(srcDevice)), ByteCount);
  return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  if (auto* real = real_symbol<decltype(&cuMemcpyDtoD_v2)>("cuMemcpyDtoD_v2"))
    return real(dstDevice, srcDevice, ByteCount);
  return cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

// Async copies are synchronous in emulation (documented in the header of
// every CUDA-compat layer with a CPU backend); the stream argument is
// accepted and ignored, like the real driver does for the NULL stream.
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
                           size_t ByteCount, CUstream /*hStream*/) {
  if (auto* real = real_symbol<decltype(&cuMemcpyHtoDAsync)>("cuMemcpyHtoDAsync"))
    return real(dstDevice, srcHost, ByteCount, nullptr);
  return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream /*hStream*/) {
  if (auto* real = real_symbol<decltype(&cuMemcpyDtoHAsync)>("cuMemcpyDtoHAsync"))
    return real(dstHost, srcDevice, ByteCount, nullptr);
  return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost,
                              size_t ByteCount, CUstream hStream) {
  if (auto* real = real_symbol<CUresult (*)(CUdeviceptr, const void*, size_t, CUstream)>(
          "cuMemcpyHtoDAsync_v2"))
    return real(dstDevice, srcHost, ByteCount, hStream);
  return cuMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream);
}

CUresult cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream) {
  if (auto* real = real_symbol<CUresult (*)(void*, CUdeviceptr, size_t, CUstream)>(
          "cuMemcpyDtoHAsync_v2"))
    return real(dstHost, srcDevice, ByteCount, hStream);
  return cuMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream);
}

CUresult cuMemGetInfo(size_t* free, size_t* total) {
  if (auto* real = real_symbol<decltype(&cuMemGetInfo)>("cuMemGetInfo"))
    return real(free, total);
  // The backend does not track consumption: total is the physical memory
  // (queried), free is reported equal to it so callers proceed and the
  // allocator itself returns OOM honestly when malloc fails.
  const size_t phys = host_phys_mem();
  if (total) *total = phys;
  if (free) *free = phys;
  if (!free && !total) return CUDA_ERROR_INVALID_VALUE;
  return CUDA_SUCCESS;
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total) {
  if (auto* real = real_symbol<decltype(&cuMemGetInfo_v2)>("cuMemGetInfo_v2"))
    return real(free, total);
  return cuMemGetInfo(free, total);
}

// Pinned host memory has no distinct meaning without DMA hardware:
// plain host allocation with identical lifetime semantics.
CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int /*Flags*/) {
  if (auto* real = real_symbol<decltype(&cuMemHostAlloc)>("cuMemHostAlloc"))
    return real(pp, bytesize, 0);
  if (!pp) return CUDA_ERROR_INVALID_VALUE;
  if (bytesize == 0) bytesize = 1;
  void* p = std::malloc(bytesize);
  if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
  *pp = p;
  return CUDA_SUCCESS;
}

CUresult cuMemFreeHost(void* p) {
  if (auto* real = real_symbol<decltype(&cuMemFreeHost)>("cuMemFreeHost")) return real(p);
  std::free(p);
  return CUDA_SUCCESS;
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int /*Flags*/) {
  if (auto* real = real_symbol<decltype(&cuMemHostGetDevicePointer)>("cuMemHostGetDevicePointer"))
    return real(pdptr, p, 0);
  if (!pdptr || !p) return CUDA_ERROR_INVALID_VALUE;
  // Unified address space in emulation: host and device pointers coincide.
  *pdptr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(p));
  return CUDA_SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
  if (auto* real = real_symbol<decltype(&cuMemFree_v2)>("cuMemFree_v2")) return real(dptr);
  return cuMemFree(dptr);
}

// ---- module globals / function attributes ---------------------------------

CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod,
                           const char* name) {
  if (auto* real = real_symbol<decltype(&cuModuleGetGlobal)>("cuModuleGetGlobal"))
    return real(dptr, bytes, hmod, name);
  if (!dptr || !name) return CUDA_ERROR_INVALID_VALUE;
  const uint64_t mkey = reinterpret_cast<uint64_t>(hmod);
  const auto mit = ctx().modules.find(mkey);
  if (mit == ctx().modules.end()) return CUDA_ERROR_INVALID_HANDLE;
  const auto& prog = mit->second;
  const auto git = prog->globalIndex.find(name);
  if (git == prog->globalIndex.end()) return CUDA_ERROR_INVALID_VALUE;
  const PtxGlobalVar& gv = prog->globals[git->second];
  const size_t nbytes =
      static_cast<size_t>(gv.elemBytes) * static_cast<size_t>(gv.count > 0 ? gv.count : 1);
  // Zero-initialised backing store, like CUDA device memory; allocated once
  // per (module, symbol) and shared with the interpreter, so host writes
  // through the returned pointer are visible to subsequently launched kernels.
  const uint64_t key = global_storage_key(mkey, git->second);
  std::vector<uint8_t>& store = ctx().globalStorage[key];
  if (store.size() != nbytes) store.assign(nbytes, 0);
  if (bytes) *bytes = nbytes;
  *dptr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(store.data()));
  return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod,
                              const char* name) {
  if (auto* real = real_symbol<decltype(&cuModuleGetGlobal_v2)>("cuModuleGetGlobal_v2"))
    return real(dptr, bytes, hmod, name);
  return cuModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) {
  if (auto* real = real_symbol<decltype(&cuFuncGetAttribute)>("cuFuncGetAttribute"))
    return real(pi, attrib, hfunc);
  if (!pi || !hfunc) return CUDA_ERROR_INVALID_VALUE;
  const auto fit = ctx().functions.find(reinterpret_cast<uint64_t>(hfunc));
  if (fit == ctx().functions.end()) return CUDA_ERROR_INVALID_HANDLE;
  const auto mit = ctx().modules.find(fit->second.moduleKey);
  if (mit == ctx().modules.end()) return CUDA_ERROR_INVALID_HANDLE;
  KernelStats st;
  if (!kernel_stats(*mit->second, fit->second.fnIndex, &st))
    return CUDA_ERROR_INVALID_VALUE;
  switch (attrib) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = 1024; return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
      *pi = static_cast<int>(st.staticSmem);
      return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_NUM_REGS:
      // Static upper bound from .reg declarations (see occupancy.h): never
      // undercounts, so occupancy math stays on the safe side.
      *pi = st.numRegs;
      return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION:
      // Compute capability the module targets (.target sm_XX); 0 when the
      // text carries no .target line (hand-written PTX without a header).
      *pi = st.targetSm;
      return CUDA_SUCCESS;
    default: return CUDA_ERROR_NOT_SUPPORTED;
  }
}

// ---- streams (synchronous backend: handles are tokens) --------------------

CUresult cuStreamCreate(CUstream* phStream, unsigned int /*Flags*/) {
  if (auto* real = real_symbol<decltype(&cuStreamCreate)>("cuStreamCreate"))
    return real(phStream, 0);
  if (!phStream) return CUDA_ERROR_INVALID_VALUE;
  *phStream = reinterpret_cast<CUstream>(new int(1));  // opaque token
  return CUDA_SUCCESS;
}

CUresult cuStreamDestroy(CUstream stream) {
  if (auto* real = real_symbol<decltype(&cuStreamDestroy)>("cuStreamDestroy"))
    return real(stream);
  delete reinterpret_cast<int*>(stream);
  return CUDA_SUCCESS;
}

CUresult cuStreamDestroy_v2(CUstream stream) {
  if (auto* real = real_symbol<decltype(&cuStreamDestroy_v2)>("cuStreamDestroy_v2"))
    return real(stream);
  return cuStreamDestroy(stream);
}

CUresult cuStreamSynchronize(CUstream /*hStream*/) {
  if (auto* real = real_symbol<decltype(&cuStreamSynchronize)>("cuStreamSynchronize"))
    return real(nullptr);
  return CUDA_SUCCESS;  // everything already completed
}

CUresult cuStreamQuery(CUstream /*hStream*/) {
  if (auto* real = real_symbol<decltype(&cuStreamQuery)>("cuStreamQuery")) return real(nullptr);
  return CUDA_SUCCESS;  // idle, like a completed NULL-stream query
}

// ---- errors ----------------------------------------------------------------

CUresult cuGetErrorName(CUresult error, const char** pStr) {
  if (auto* real = real_symbol<decltype(&cuGetErrorName)>("cuGetErrorName"))
    return real(error, pStr);
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  switch (error) {
    case CUDA_SUCCESS: *pStr = "CUDA_SUCCESS"; return CUDA_SUCCESS;
    case CUDA_ERROR_INVALID_VALUE: *pStr = "CUDA_ERROR_INVALID_VALUE"; return CUDA_SUCCESS;
    case CUDA_ERROR_OUT_OF_MEMORY: *pStr = "CUDA_ERROR_OUT_OF_MEMORY"; return CUDA_SUCCESS;
    case CUDA_ERROR_NOT_INITIALIZED: *pStr = "CUDA_ERROR_NOT_INITIALIZED"; return CUDA_SUCCESS;
    case CUDA_ERROR_INVALID_HANDLE: *pStr = "CUDA_ERROR_INVALID_HANDLE"; return CUDA_SUCCESS;
    case CUDA_ERROR_NOT_SUPPORTED: *pStr = "CUDA_ERROR_NOT_SUPPORTED"; return CUDA_SUCCESS;
    default: *pStr = "CUDA_ERROR_UNKNOWN"; return CUDA_SUCCESS;
  }
}

// ---- occupancy ---------------------------------------------------------------
// CUfunction -> (модуль, индекс) -> KernelStats -> тот же калькулятор, что у
// Runtime API. Невозможная конфигурация — успех с нулём (как настоящий драйвер).

namespace {

bool stats_for_cufunc(CUfunction func, KernelStats* st) {
  if (func == nullptr || st == nullptr) return false;
  const auto fit = ctx().functions.find(reinterpret_cast<uint64_t>(func));
  if (fit == ctx().functions.end()) return false;
  const auto mit = ctx().modules.find(fit->second.moduleKey);
  if (mit == ctx().modules.end()) return false;
  return kernel_stats(*mit->second, fit->second.fnIndex, st);
}

}  // namespace

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func,
                                                     int blockSize, size_t dynamicSMemSize) {
  if (auto* real =
          real_symbol<decltype(&cuOccupancyMaxActiveBlocksPerMultiprocessor)>(
              "cuOccupancyMaxActiveBlocksPerMultiprocessor"))
    return real(numBlocks, func, blockSize, dynamicSMemSize);
  if (!numBlocks || !func || blockSize <= 0) return CUDA_ERROR_INVALID_VALUE;
  KernelStats st;
  if (!stats_for_cufunc(func, &st)) return CUDA_ERROR_INVALID_HANDLE;
  *numBlocks = occ_active_blocks(st, blockSize, dynamicSMemSize);
  return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func,
                                                               int blockSize,
                                                               size_t dynamicSMemSize,
                                                               unsigned int flags) {
  if (auto* real =
          real_symbol<decltype(&cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)>(
              "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"))
    return real(numBlocks, func, blockSize, dynamicSMemSize, flags);
  // Только документированные флаги; на sm_75 флаг кэширования на лимиты не
  // влияет (partitioned GC там не поддерживается — см. occupancy.h).
  if (flags != 0 && flags != 1) return CUDA_ERROR_INVALID_VALUE;
  return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize,
                                                     dynamicSMemSize);
}

CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func,
                                          CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                          size_t dynamicSMemSize, int blockSizeLimit) {
  if (auto* real = real_symbol<decltype(&cuOccupancyMaxPotentialBlockSize)>(
          "cuOccupancyMaxPotentialBlockSize"))
    return real(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize,
                blockSizeLimit);
  if (!minGridSize || !blockSize || !func) return CUDA_ERROR_INVALID_VALUE;
  KernelStats st;
  if (!stats_for_cufunc(func, &st)) return CUDA_ERROR_INVALID_HANDLE;
  // blockSizeLimit <= 0 означает «без лимита» (как у настоящего драйвера).
  if (blockSizeLimit > 0 && blockSizeLimit < 1024) {
    // Тот же перебор configurator'а, но с обрезанным верхом.
    int bestBlock = 0, bestNum = 0, bestThreads = 0;
    for (int trySize = (blockSizeLimit / 32) * 32; trySize > 0; trySize -= 32) {
      size_t dyn = dynamicSMemSize;
      if (blockSizeToDynamicSMemSize != nullptr) dyn = blockSizeToDynamicSMemSize(trySize);
      const int active = occ_active_blocks(st, trySize, dyn);
      if (trySize * active > bestThreads) {
        bestThreads = trySize * active;
        bestBlock = trySize;
        bestNum = active;
      }
      if (bestThreads == 1024) break;
    }
    *blockSize = bestBlock;
    *minGridSize = bestNum;  // один SM у эмулированного устройства
    return CUDA_SUCCESS;
  }
  occ_potential_block_size(minGridSize, blockSize, st, blockSizeToDynamicSMemSize,
                           dynamicSMemSize, /*numSms=*/1);
  return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char** pStr) {
  if (auto* real = real_symbol<decltype(&cuGetErrorString)>("cuGetErrorString"))
    return real(error, pStr);
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  static const char* kUnknown = "unknown driver error";
  static const char* kMessages[] = {"CUDA_SUCCESS", "invalid value", "out of memory",
                                    "not initialized"};
  if (error >= 0 && error < 4) *pStr = kMessages[error];
  else if (error == CUDA_ERROR_INVALID_HANDLE) *pStr = "invalid resource handle";
  else if (error == CUDA_ERROR_NOT_SUPPORTED) *pStr = "not supported on this device/backend";
  else *pStr = kUnknown;
  return CUDA_SUCCESS;
}

}  // extern "C"