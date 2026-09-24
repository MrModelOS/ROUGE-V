// atom_cas (cas/exch/min/max) via the CUDA Driver API on the
// CPU-emulated ROUGE device — the interpreter fallback path.

#include <rouge/cuda.h>

#include "atom_cas_ptx.h"

#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK(expr)                                                        \
  do {                                                                     \
    const CUresult _r = (expr);                                            \
    if (_r != CUDA_SUCCESS) {                                              \
      const char* _s = "?";                                                \
      cuGetErrorString(_r, &_s);                                           \
      std::fprintf(stderr, "%s:%d: %s -> %s (%d)\n", __FILE__, __LINE__,   \
                   #expr, _s, static_cast<int>(_r));                       \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main() {
  const uint32_t N = 4096;
  const int BX = 256;
  const int GX = static_cast<int>(N) / BX;

  std::vector<uint32_t> cas_data(N, 42), exch_data(N, 11), min_data(N, 50), max_data(N, 20);
  std::vector<uint32_t> cas_old(N, 0), exch_old(N, 0), min_old(N, 0), max_old(N, 0);

  CHECK(cuInit(0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, 0));

  CUdeviceptr d_cas = 0, d_exch = 0, d_min = 0, d_max = 0;
  CUdeviceptr d_cas_old = 0, d_exch_old = 0, d_min_old = 0, d_max_old = 0;
  CHECK(cuMemAlloc(&d_cas, N * sizeof(uint32_t)));
  CHECK(cuMemAlloc(&d_exch, N * sizeof(uint32_t)));
  CHECK(cuMemAlloc(&d_min, N * sizeof(uint32_t)));
  CHECK(cuMemAlloc(&d_max, N * sizeof(uint32_t)));
  CHECK(cuMemAlloc(&d_cas_old, N * sizeof(uint32_t)));
  CHECK(cuMemAlloc(&d_exch_old, N * sizeof(uint32_t)));
  CHECK(cuMemAlloc(&d_min_old, N * sizeof(uint32_t)));
  CHECK(cuMemAlloc(&d_max_old, N * sizeof(uint32_t)));
  CHECK(cuMemcpyHtoD(d_cas, cas_data.data(), N * sizeof(uint32_t)));
  CHECK(cuMemcpyHtoD(d_exch, exch_data.data(), N * sizeof(uint32_t)));
  CHECK(cuMemcpyHtoD(d_min, min_data.data(), N * sizeof(uint32_t)));
  CHECK(cuMemcpyHtoD(d_max, max_data.data(), N * sizeof(uint32_t)));
  const uint32_t zero = 0;
  CHECK(cuMemcpyHtoD(d_cas_old, &zero, sizeof(uint32_t)));
  // broadcast zero to all (simple: copy from host zeroed arrays)
  CHECK(cuMemcpyHtoD(d_cas_old, cas_old.data(), N * sizeof(uint32_t)));
  CHECK(cuMemcpyHtoD(d_exch_old, exch_old.data(), N * sizeof(uint32_t)));
  CHECK(cuMemcpyHtoD(d_min_old, min_old.data(), N * sizeof(uint32_t)));
  CHECK(cuMemcpyHtoD(d_max_old, max_old.data(), N * sizeof(uint32_t)));

  CUmodule mod = nullptr;
  CHECK(cuModuleLoadData(&mod, kAtomCasPtx));
  CUfunction fn = nullptr;
  CHECK(cuModuleGetFunction(&fn, mod, "atom_cas"));

  const unsigned int n = N;
  void* params[9] = {&d_cas, &d_exch, &d_min, &d_max, &d_cas_old, &d_exch_old, &d_min_old, &d_max_old, const_cast<unsigned int*>(&n)};
  CHECK(cuLaunchKernel(fn, GX, 1, 1, BX, 1, 1, 0, nullptr, params, nullptr));
  CHECK(cuCtxSynchronize());

  CHECK(cuMemcpyDtoH(cas_data.data(), d_cas, N * sizeof(uint32_t)));
  CHECK(cuMemcpyDtoH(exch_data.data(), d_exch, N * sizeof(uint32_t)));
  CHECK(cuMemcpyDtoH(min_data.data(), d_min, N * sizeof(uint32_t)));
  CHECK(cuMemcpyDtoH(max_data.data(), d_max, N * sizeof(uint32_t)));
  CHECK(cuMemcpyDtoH(cas_old.data(), d_cas_old, N * sizeof(uint32_t)));
  CHECK(cuMemcpyDtoH(exch_old.data(), d_exch_old, N * sizeof(uint32_t)));
  CHECK(cuMemcpyDtoH(min_old.data(), d_min_old, N * sizeof(uint32_t)));
  CHECK(cuMemcpyDtoH(max_old.data(), d_max_old, N * sizeof(uint32_t)));

  int bad = 0;
  for (uint32_t i = 0; i < N; ++i) if (cas_old[i] != 42) { std::fprintf(stderr, "cas_old[%u]=%u want 42\n", i, cas_old[i]); bad=1; break; }
  for (uint32_t i = 0; i < N; ++i) if (cas_data[i] != 100) { std::fprintf(stderr, "cas_data[%u]=%u want 100\n", i, cas_data[i]); bad=1; break; }
  for (uint32_t i = 0; i < N; ++i) if (exch_old[i] != 11) { std::fprintf(stderr, "exch_old[%u]=%u want 11\n", i, exch_old[i]); bad=1; break; }
  for (uint32_t i = 0; i < N; ++i) if (exch_data[i] != 55) { std::fprintf(stderr, "exch_data[%u]=%u want 55\n", i, exch_data[i]); bad=1; break; }
  for (uint32_t i = 0; i < N; ++i) if (min_old[i] != 50) { std::fprintf(stderr, "min_old[%u]=%u want 50\n", i, min_old[i]); bad=1; break; }
  for (uint32_t i = 0; i < N; ++i) if (min_data[i] != 10) { std::fprintf(stderr, "min_data[%u]=%u want 10\n", i, min_data[i]); bad=1; break; }
  for (uint32_t i = 0; i < N; ++i) if (max_old[i] != 20) { std::fprintf(stderr, "max_old[%u]=%u want 20\n", i, max_old[i]); bad=1; break; }
  for (uint32_t i = 0; i < N; ++i) if (max_data[i] != 90) { std::fprintf(stderr, "max_data[%u]=%u want 90\n", i, max_data[i]); bad=1; break; }

  if (bad) {
    std::fprintf(stderr, "FAIL: driver-API atom_cas\n");
    return 1;
  }
  std::printf("PASS: driver-API atom_cas (cas/exch/min/max) on CPU-emulated ROUGE device: %d blocks x %d\n", GX, BX);

  CHECK(cuMemFree(d_cas));
  CHECK(cuMemFree(d_exch));
  CHECK(cuMemFree(d_min));
  CHECK(cuMemFree(d_max));
  CHECK(cuMemFree(d_cas_old));
  CHECK(cuMemFree(d_exch_old));
  CHECK(cuMemFree(d_min_old));
  CHECK(cuMemFree(d_max_old));
  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));
  return 0;
}
