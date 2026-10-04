// Волна 1, вторая программа: широкий набор Runtime API.
//
// Проверяет ровно те символы, которых не хватало программе из статьи:
// cudaSetDevice / cudaGetDevice / cudaGetDeviceProperties / cudaMallocHost /
// cudaHostAlloc / cudaFreeHost / cudaMemcpyAsync / cudaMemset /
// cudaPeekAtLastError / cudaStreamCreateWithFlags / cudaDeviceReset.
//
// Печатает значения, которые читаются из cudaDeviceProp, — раскладка этой
// структуры сравнивается с настоящей в отдельном тесте на offsets.
#include <cstdio>

__global__ void bump(float* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += 1.0f;
}

int main() {
  int dev = -1;
  if (cudaSetDevice(0) != cudaSuccess) return 2;
  if (cudaGetDevice(&dev) != cudaSuccess) return 2;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 2;

  const int N = 256;
  float* host = nullptr;
  if (cudaMallocHost(&host, N * sizeof(float)) != cudaSuccess) return 2;
  float* pinned = nullptr;
  if (cudaHostAlloc(&pinned, 64, cudaHostAllocDefault) != cudaSuccess) return 2;
  float* d = nullptr;
  if (cudaMalloc(&d, N * sizeof(float)) != cudaSuccess) return 2;
  for (int i = 0; i < N; ++i) host[i] = (float)i;

  cudaStream_t stream;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(d, host, N * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    return 2;
  // Ordered on purpose: the copy back must not race the kernel.
  cudaStreamSynchronize(stream);
  bump<<<1, N>>>(d, N);
  if (cudaMemcpyAsync(host, d, N * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    return 2;
  if (cudaStreamSynchronize(stream) != cudaSuccess) return 2;
  if (cudaDeviceSynchronize() != cudaSuccess) return 2;
  if (cudaPeekAtLastError() != cudaSuccess) return 2;
  if (cudaMemset(d, 0, 16) != cudaSuccess) return 2;

  const bool ok = (host[3] == 4.0f);
  std::printf("%s dev=%d sm=%d.%d warp=%d name=%s err=%s\n", ok ? "OK" : "BAD", dev, prop.major,
              prop.minor, prop.warpSize, prop.name, cudaGetErrorString(cudaGetLastError()));

  cudaFreeHost(host);
  cudaFreeHost(pinned);
  cudaFree(d);
  cudaStreamDestroy(stream);
  cudaDeviceReset();
  return ok ? 0 : 1;
}