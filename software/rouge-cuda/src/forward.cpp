#include "rouge_internal.h"

#include <dlfcn.h>

namespace rouge {

Context& ctx() {
  static Context instance;
  return instance;
}

void* load_real_symbol(const char* name) {
  // Probe once for a real CUDA library already loaded in the process.
  // If present, we are in "interposition" mode and forward to it.
  static void* handle = []() -> void* {
    for (const char* lib : {"libcuda.so.1", "libcuda.so", "libcudart.so.12", "libcudart.so"}) {
      if (void* h = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD)) return h;
    }
    return nullptr;
  }();
  if (!handle) return nullptr;
  return dlsym(handle, name);
}

}  // namespace rouge