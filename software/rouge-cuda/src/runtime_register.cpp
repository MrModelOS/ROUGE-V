// Runtime-API launch path: what nvcc's <<<>>> actually calls.
//
// Probe (objdump of an object built by nvcc 13.4 -arch=sm_75 -c) shows the
// real contract, so nothing here is guessed from memory:
//
//   __cudaRegisterFatBinary / __cudaRegisterFatBinaryEnd / __cudaUnregisterFatBinary
//   __cudaRegisterFunction   rdi = fatbin handle
//                            rsi = ADDRESS OF the kernel's stub variable
//                            rdx = mangled kernel name (".rodata")
//   __cudaPushCallConfiguration / __cudaPopCallConfiguration
//   __cudaLaunchKernel       rdi = stub, rsi+rdx = gridDim, rcx+r8 = blockDim,
//                            r9 = args, stack = sharedMemBytes, stream
//
// A kernel's stub is the value stored through rsi; __cudaLaunchKernel gets
// that same value, so stub -> name is the whole dispatch table a runtime needs.
// The PTX itself comes from the module list named by ROUGE_CUDA_PTX (colon
// separated .ptx paths); fatbin parsing is the next milestone, so until then
// the PTX is supplied out of band rather than read out of the binary.

#include <rouge/cuda.h>
#include <rouge/cuda_runtime_api.h>

#include "fatbin.h"
#include "ptx.h"
#include "rouge_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// The launch configuration pushed by the <<<>>> sugar, popped by the launch.
struct CallConfig {
  dim3 grid{1, 1, 1};
  dim3 block{1, 1, 1};
  size_t sharedMem = 0;
  cudaStream_t stream = nullptr;
};

std::vector<CallConfig>& config_stack() {
  static std::vector<CallConfig> stack;
  return stack;
}

// Modules named by ROUGE_CUDA_PTX, parsed on first use. (Fatbin modules land
// in the shared runtime_modules() registry straight from __cudaRegisterFatBinary;
// stub_names() lives in rouge_internal.h so every TU shares one table.)

bool load_ptx_modules() {
  const char* list = std::getenv("ROUGE_CUDA_PTX");
  if (list == nullptr || *list == '\0') {
    std::fprintf(stderr,
                 "[ROUGE-CUDA] no kernel PTX: set ROUGE_CUDA_PTX=<file.ptx[:file.ptx...]>\n");
    return false;
  }
  const std::string all(list);
  size_t pos = 0;
  while (pos <= all.size()) {
    const size_t colon = all.find(':', pos);
    const std::string path = all.substr(pos, colon == std::string::npos ? std::string::npos
                                                                        : colon - pos);
    if (!path.empty()) {
      FILE* f = std::fopen(path.c_str(), "rb");
      if (f == nullptr) {
        std::fprintf(stderr, "[ROUGE-CUDA] cannot open %s\n", path.c_str());
        return false;
      }
      std::fseek(f, 0, SEEK_END);
      const long size = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      std::string text(static_cast<size_t>(size < 0 ? 0 : size), '\0');
      if (size > 0 && std::fread(&text[0], 1, text.size(), f) != text.size()) {
        std::fclose(f);
        std::fprintf(stderr, "[ROUGE-CUDA] short read on %s\n", path.c_str());
        return false;
      }
      std::fclose(f);
      std::string err;
      auto prog = rouge::parse_ptx(text, &err);
      if (!prog) {
        std::fprintf(stderr, "[ROUGE-CUDA] PTX parse failed for %s: %s\n", path.c_str(),
                     err.c_str());
        return false;
      }
      rouge::runtime_modules().push_back(std::shared_ptr<rouge::PtxProgram>(std::move(prog)));
    }
    if (colon == std::string::npos) break;
    pos = colon + 1;
  }
  return !rouge::runtime_modules().empty();
}

// Find a kernel by mangled name across the loaded modules.
bool find_kernel(const std::string& name, rouge::PtxProgram** out, int* fnIndex) {
  for (auto& mod : rouge::runtime_modules()) {
    const auto it = mod->functionIndex.find(name);
    if (it != mod->functionIndex.end()) {
      *out = mod.get();
      *fnIndex = it->second;
      return true;
    }
  }
  return false;
}

}  // namespace

namespace rouge {
// Defined here rather than in driver_api.cpp: the registration tables belong
// to the Runtime API path, not to the Driver API.
cudaError_t launch_registered(const void* stub, dim3 grid, dim3 block, void** args,
                              size_t sharedMemBytes, cudaStream_t stream) {
  (void)stream;  // CPU backend is synchronous
  if (stub == nullptr || args == nullptr) return cudaErrorInvalidValue;
  const auto it = stub_names().find(reinterpret_cast<uintptr_t>(stub));
  if (it == stub_names().end()) {
    std::fprintf(stderr, "[ROUGE-CUDA] kernel stub %p was never registered\n", stub);
    return cudaErrorInvalidValue;
  }
  if (rouge::runtime_modules().empty() && !load_ptx_modules()) return cudaErrorInvalidValue;

  rouge::PtxProgram* prog = nullptr;
  int fnIndex = -1;
  if (!find_kernel(it->second, &prog, &fnIndex)) {
    std::fprintf(stderr, "[ROUGE-CUDA] no PTX entry named %s\n", it->second.c_str());
    return cudaErrorInvalidValue;
  }
  const rouge::PtxFunction& fn = prog->functions[fnIndex];
  std::vector<uint8_t> blob(static_cast<size_t>(fn.paramsBlobSize), 0);
  for (const auto& p : fn.params) {
    if (p.index < 0 || !args[p.index] || p.offset + p.width > static_cast<int>(blob.size()))
      return cudaErrorInvalidValue;
    std::memcpy(blob.data() + p.offset, args[p.index], static_cast<size_t>(p.width));
  }
  std::string err;
  if (!rouge::execute_kernel(*prog, fnIndex, blob, grid.x, grid.y, grid.z,
                             block.x, block.y, block.z, &err))
    return cudaErrorInvalidValue;
  if (sharedMemBytes != 0) {
    std::fprintf(stderr,
                 "[ROUGE-CUDA] dynamic .shared (%zu bytes) is not supported by the "
                 "interpreter\n",
                 sharedMemBytes);
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}
}  // namespace rouge

extern "C" {

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                             size_t sharedMemBytes, cudaStream_t stream) {
  return rouge::launch_registered(func, gridDim, blockDim, args, sharedMemBytes, stream);
}

// ---- fatbin registration ---------------------------------------------------
// nvcc's static initialisers call these before main. The protocol (verified on
// an object built by nvcc 13.4, see the file header and the disassembly of
// __nv_save_fatbinhandle_for_managed_rt) is:
//
//   __cudaRegisterFatBinary(handle)
//        handle points at a struct whose FIRST FIELD is the module's
//        __nv_cudaEntityRegisterCallback; the runtime must call it with the
//        handle, and that callback is what walks the module and calls
//        __cudaRegisterFunction once per kernel.
//
// Skipping that call means no kernel is ever registered and every launch is a
// silent no-op, so the callback invocation is the whole registration protocol.
// Parsing the fatbin image itself (ELF + compressed PTX) is the next milestone;
// until then the PTX comes from ROUGE_CUDA_PTX.

// Registration protocol, verified by disassembling an executable linked from
// `nvcc -arch=sm_75 -c` output:
//
//   .init_array -> __sti____cudaRegisterAll():
//       handle = __cudaRegisterFatBinary(&__TMC_END__);   // rax IS the handle
//       __nv_cudaEntityRegisterCallback(handle);          // program's own code
//       __cudaRegisterFatBinaryEnd(&handle);
//   __nv_cudaEntityRegisterCallback(handle) then calls, per kernel,
//       __cudaRegisterFunction(handle, stub, name, traits, ...)
//   with the STUB ADDRESS itself as the second argument (not a pointer to it).
//
// So the runtime never has to parse the fatbin to learn kernel names: the
// module's own registration callback reports them. What the runtime does own is
// turning a name into executable code, which for now comes from
// ROUGE_CUDA_PTX; extracting PTX out of the fatbin image is the next milestone
// and needs no protocol change.

void* __cudaRegisterFatBinary(const void* fatCubinHandle) {
  // The handle travels back into the program and is handed to its registration
  // callback, which stores it — it must be non-null and stable.
  //
  // The same handle points at the fatbin descriptor, so this is our moment to
  // take the PTX straight out of the binary: after this the program needs no
  // ROUGE_CUDA_PTX and is self-sufficient, exactly like with a real driver.
  if (fatCubinHandle != nullptr) {
    std::string err;
    const std::vector<std::string> texts = rouge_fatbin_extract_ptx(fatCubinHandle, &err);
    for (const std::string& text : texts) {
      std::string perr;
      auto prog = rouge::parse_ptx(text, &perr);
      if (!prog) {
        std::fprintf(stderr, "[ROUGE-CUDA] fatbin PTX parse failed: %s\n", perr.c_str());
        continue;
      }
      rouge::runtime_modules().push_back(std::shared_ptr<rouge::PtxProgram>(std::move(prog)));
    }
    if (!texts.empty()) {
      std::fprintf(stderr, "[ROUGE-CUDA] %zu PTX module(s) taken from the fatbin\n",
                   texts.size());
    } else {
      // Loud, specific reason: a silent empty result would make a broken
      // fatbin look like a missing ROUGE_CUDA_PTX much later on.
      std::fprintf(stderr, "[ROUGE-CUDA] fatbin PTX unavailable: %s\n", err.c_str());
    }
  }
  return const_cast<void*>(fatCubinHandle);
}

void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) { (void)fatCubinHandle; }

void __cudaUnregisterFatBinary(void* fatCubinHandle) { (void)fatCubinHandle; }

// stub is the kernel's stub function ADDRESS; __cudaLaunchKernel later receives
// exactly that value, so stub -> name is the whole dispatch table.
void __cudaRegisterFunction(const void* fatCubinHandle, void* stub, const char* name,
                            const char* traits, const unsigned int* funcAnnotations,
                            const unsigned int* deviceAnnotations, ...) {
  (void)fatCubinHandle;
  (void)traits;
  (void)funcAnnotations;
  (void)deviceAnnotations;
  if (stub == nullptr || name == nullptr) return;
  rouge::stub_names()[reinterpret_cast<uintptr_t>(stub)] = name;
}

void __cudaUnregisterFunction(const void* func) {
  rouge::stub_names().erase(reinterpret_cast<uintptr_t>(func));
}

cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMemBytes,
                                        cudaStream_t stream) {
  config_stack().push_back(CallConfig{gridDim, blockDim, sharedMemBytes, stream});
  return cudaSuccess;
}

cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim,
                                       size_t* sharedMemBytes, cudaStream_t* stream) {
  if (config_stack().empty()) return cudaErrorInvalidValue;
  const CallConfig cfg = config_stack().back();
  config_stack().pop_back();
  if (gridDim) *gridDim = cfg.grid;
  if (blockDim) *blockDim = cfg.block;
  if (sharedMemBytes) *sharedMemBytes = cfg.sharedMem;
  if (stream) *stream = cfg.stream;
  return cudaSuccess;
}

cudaError_t __cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                               size_t sharedMemBytes, cudaStream_t stream) {
  return rouge::launch_registered(func, gridDim, blockDim, args, sharedMemBytes, stream);
}

// __cudaGetKernel is called once per kernel from the device stub, and the value
// it writes through the handle is what __cudaLaunchKernel later receives as its
// first argument (verified in the stub's disassembly). Writing the stub address
// itself keeps stub -> name lookup valid and avoids inventing a second table.
cudaError_t __cudaGetKernel(void** handle, const void* func) {
  if (handle == nullptr || func == nullptr) return cudaErrorInvalidValue;
  *handle = const_cast<void*>(func);
  return cudaSuccess;
}

}  // extern "C"