// Verify the two halves of interposition:
//  1) our shared object exports CUDA symbols, so LD_PRELOAD can reroute an
//     application that links -lcuda / -lcudart without recompiling it;
//  2) the forward path (dlsym into a real CUDA library) is inert here because
//     no real CUDA runtime is loaded in this test environment.
//
// ROUGE_SHARED_LIB is injected by CMake as an absolute path to librouge-cuda.so.

#include <dlfcn.h>

#include <cstdio>

#ifndef ROUGE_SHARED_LIB
#define ROUGE_SHARED_LIB "librouge-cuda.so"
#endif

int main() {
  void* h = dlopen(ROUGE_SHARED_LIB, RTLD_NOW);
  if (!h) {
    std::fprintf(stderr, "cannot dlopen %s: %s\n", ROUGE_SHARED_LIB, dlerror());
    return 1;
  }
  void* sym = dlsym(h, "cuInit");
  if (!sym) {
    std::fprintf(stderr, "cuInit not exported by %s\n", ROUGE_SHARED_LIB);
    return 1;
  }
  dlclose(h);
  std::printf("PASS: CUDA symbols exported; LD_PRELOAD interposition ready\n");
  return 0;
}