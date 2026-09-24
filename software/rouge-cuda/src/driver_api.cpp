#include <rouge/cuda.h>

#include "ptx.h"
#include "rouge_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

CUresult cuCtxCreate(CUcontext* pctx, unsigned int /*flags*/, int /*dev*/) {
  if (auto* real = real_symbol<decltype(&cuCtxCreate)>("cuCtxCreate")) return real(pctx, 0, 0);
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = reinterpret_cast<CUcontext>(&ctx());  // dummy but stable
  return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
  if (auto* real = real_symbol<decltype(&cuCtxSynchronize)>("cuCtxSynchronize")) return real();
  return CUDA_SUCCESS;  // everything is synchronous in CPU emulation
}

CUresult cuCtxDestroy(CUcontext /*ctx*/) {
  if (auto* real = real_symbol<decltype(&cuCtxDestroy)>("cuCtxDestroy")) return real(nullptr);
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
  if (ctx().modules.erase(key) == 0) return CUDA_ERROR_INVALID_VALUE;
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
                        unsigned int /*sharedMemBytes*/, void* /*hStream*/,
                        void** kernelParams, void** /*extra*/) {
  if (auto* real = real_symbol<decltype(&cuLaunchKernel)>("cuLaunchKernel"))
    return real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                0, nullptr, kernelParams, nullptr);

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
  if (!execute_kernel(*prog, fnIndex, blob, gridDimX, gridDimY, gridDimZ,
                      blockDimX, blockDimY, blockDimZ, &err)) {
    std::fprintf(stderr, "[ROUGE-CUDA] kernel launch failed: %s\n", err.c_str());
    return CUDA_ERROR_INVALID_VALUE;
  }
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
  else *pStr = kUnknown;
  return CUDA_SUCCESS;
}

}  // extern "C"