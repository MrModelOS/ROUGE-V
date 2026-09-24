// Runtime-API pipeline: malloc -> memcpy H2D -> D2D -> D2H -> compare.
// All pointers are host addresses in CPU emulation, so kinds are transparent.

#include <rouge/cuda_runtime_api.h>

#include <cstdio>
#include <vector>

#define CHECK(expr)                                                              \
  do {                                                                           \
    const cudaError_t _e = (expr);                                               \
    if (_e != cudaSuccess) {                                                     \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,       \
                   cudaGetErrorString(_e));                                      \
      return 1;                                                                  \
    }                                                                            \
  } while (0)

int main() {
  const int N = 1024;
  std::vector<int> host(N), out(N, 0);
  for (int i = 0; i < N; ++i) host[i] = i * 3;

  void* d1 = nullptr;
  void* d2 = nullptr;
  void* d3 = nullptr;
  CHECK(cudaMalloc(&d1, N * sizeof(int)));
  CHECK(cudaMalloc(&d2, N * sizeof(int)));
  CHECK(cudaMalloc(&d3, N * sizeof(int)));

  CHECK(cudaMemcpy(d1, host.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d2, d1, N * sizeof(int), cudaMemcpyDeviceToDevice));
  CHECK(cudaMemcpy(d3, d2, N * sizeof(int), cudaMemcpyDeviceToDevice));
  CHECK(cudaMemcpy(out.data(), d3, N * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < N; ++i) {
    if (out[i] != host[i]) {
      std::fprintf(stderr, "mismatch at %d: got %d, expected %d\n", i, out[i], host[i]);
      return 1;
    }
  }

  cudaStream_t s = nullptr;
  CHECK(cudaStreamCreate(&s));
  CHECK(cudaStreamSynchronize(s));
  CHECK(cudaStreamDestroy(s));

  CHECK(cudaFree(d1));
  CHECK(cudaFree(d2));
  CHECK(cudaFree(d3));

  std::printf("PASS: runtime-API memory copy pipeline + streams\n");
  return 0;
}