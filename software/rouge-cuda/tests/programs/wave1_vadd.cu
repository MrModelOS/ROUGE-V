// Волна 1: программа-жертва для теста foreign_program.
//
// Скомпилирована НАСТОЯЩИМ nvcc, запускается нашим шимом без libcudart.
// Использует сахар <<<>>>, то есть весь путь регистрации ядра:
// fatbin -> callback -> __cudaRegisterFunction -> push/pop конфигурации ->
// __cudaLaunchKernel. Печатает "OK" только если все элементы сошлись.
#include <cstdio>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void scale(float* a, float k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = a[i] * k;
}

int main() {
  const int N = 4096;
  float* ha = new float[N];
  float* hb = new float[N];
  float* hc = new float[N];
  for (int i = 0; i < N; ++i) {
    ha[i] = (float)i * 0.25f;
    hb[i] = (float)(i * 2) - 0.5f;
    hc[i] = 0.0f;
  }

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  if (cudaMalloc(&da, N * sizeof(float)) != cudaSuccess) return 2;
  if (cudaMalloc(&db, N * sizeof(float)) != cudaSuccess) return 2;
  if (cudaMalloc(&dc, N * sizeof(float)) != cudaSuccess) return 2;
  if (cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
  if (cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
  if (cudaMemcpy(dc, hc, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 2;

  // Two different kernels in one module, two different launch geometries:
  // the register table must dispatch by stub, not by order.
  vadd<<<(N + 255) / 256, 256>>>(da, db, dc, N);
  scale<<<(N + 127) / 128, 128>>>(dc, 2.0f, N);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("sync failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 2;
  }

  if (cudaMemcpy(hc, dc, N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;

  int bad = 0;
  for (int i = 0; i < N; ++i) {
    const float want = (ha[i] + hb[i]) * 2.0f;
    if (hc[i] != want) {
      if (bad < 3) std::printf("i=%d got=%f want=%f\n", i, hc[i], want);
      ++bad;
    }
  }
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  if (bad != 0) {
    std::printf("BAD %d\n", bad);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}