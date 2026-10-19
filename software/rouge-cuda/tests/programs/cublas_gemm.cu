// Волна 1, третья программа: настоящая CUDA-программа, использующая
// настоящий cuBLAS.
//
// Отличие от предыдущих двух программ: здесь нет ни одного CUDA-ядра —
// только вызов cuBLAS. Проверяется подмена ПРОПРИЕТАРНОЙ библиотеки NVIDIA,
// а не только рантайма: тот же объект линкуется то с настоящей libcublas
// (эталонный прогон на карте), то с нашим шим cublas (исполнение на CPU, без
// NVIDIA вообще).
//
// Раскладка column-major (как в cuBLAS): A — m×k, B — k×n, C — m×n.
// Печать с ограниченной точностью: GPU-cuBLAS и наш CPU-GEMM складывают в
// другом порядке, и младшие биты могут отличаться — сравниваем математику,
// а не порядок сложения.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>

int main() {
  const int m = 5, n = 4, k = 6;
  float hA[m * k], hB[k * n], hC[m * n];
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < m; ++i) hA[j * m + i] = (float)(i * 7 + j * 3 + 1) * 0.25f;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < k; ++i) hB[j * k + i] = (float)(i * 2 - j * 5 + 3) * 0.5f;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) hC[j * m + i] = 0.0f;

  if (cudaSetDevice(0) != cudaSuccess) return 2;
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (cudaMalloc(&dA, sizeof(hA)) != cudaSuccess) return 2;
  if (cudaMalloc(&dB, sizeof(hB)) != cudaSuccess) return 2;
  if (cudaMalloc(&dC, sizeof(hC)) != cudaSuccess) return 2;
  if (cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
  if (cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice) != cudaSuccess) return 2;

  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 2;
  const float alpha = 1.0f, beta = 0.0f;
  if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k,
                  &beta, dC, m) != CUBLAS_STATUS_SUCCESS)
    return 2;
  if (cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;
  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  double sum = 0.0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      std::printf("c[%d][%d]=%.4f\n", i, j, (double)hC[j * m + i]);
      sum += (double)hC[j * m + i];
    }
  std::printf("sum=%.4f\n", sum);
  std::printf("OK\n");
  return 0;
}
