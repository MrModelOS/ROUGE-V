#include "rouge_internal.h"

#include <dlfcn.h>

#include <cstdlib>

namespace rouge {

Context& ctx() {
  static Context instance;
  return instance;
}

}  // namespace rouge

// Definition of the sentinel declared above. Plain definition (no extern
// keyword, so no -Wextern-initializer): C language linkage is inherited
// from the prior extern "C" declaration, keeping the dynamic symbol
// exactly "rouge_cuda_emulation_sentinel" (verified with nm -D).
int rouge_cuda_emulation_sentinel = 1;

namespace rouge {

namespace {

bool force_emu() {
  static bool v = []() {
    const char* e = std::getenv("ROUGE_CUDA_FORCE_EMU");
    return e != nullptr && *e != '\0' && *e != '0';
  }();
  return v;
}

}  // namespace

void* load_real_symbol(const char* name) {
  // ROUGE_CUDA_FORCE_EMU=1 disables forwarding entirely: every call is
  // served by the CPU-emulation backend even when a real CUDA library is
  // present. This is the auditable "zero NVIDIA calls" mode for demos.
  if (force_emu()) return nullptr;
  // Probe once for a real CUDA library already loaded in the process.
  // If present, we are in "interposition" mode and forward to it.
  static void* handle = []() -> void* {
    for (const char* lib : {"libcuda.so.1", "libcuda.so", "libcudart.so.12", "libcudart.so"}) {
      void* h = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD);
      if (!h) continue;
      // The imposter is ourselves: never forward into our own symbols.
      if (dlsym(h, "rouge_cuda_emulation_sentinel")) continue;
      return h;
    }
    return nullptr;
  }();
  if (!handle) return nullptr;
  return dlsym(handle, name);
}

}  // namespace rouge