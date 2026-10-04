// Волна 1: occupancy и атрибуты функций через оба стека.
//
// Та же схема, что у wave1_vadd/wave1_api: один исходник собирается nvcc
// против настоящего cudart и g++ против нашего шима, выводы сравниваются
// побайтово (кроме имени устройства и minGridSize — топология машин разная:
// 14 SM против 1 эмулированного; это нормализуется в сравнении).
//
// Ядро — средней тяжести (shared 1K, десятки регистров): occupancy обязана
// совпасть именно там, где регистры не упираются в границу бакета.

#include <cstdio>

__global__ void occmid(const float* a, float* b, int n) {
  __shared__ float s[256];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = (i < n) ? a[i] : 0.0f;
  s[threadIdx.x] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int k = 0; k < 256; ++k) t += s[k];
    b[blockIdx.x] = t;
  }
}

__global__ void occtiny(float* a) {
  int i = threadIdx.x;
  a[i] += 1.0f;
}

static int failures = 0;

#define CHECK_OP(expr, label)                                                     \
  do {                                                                            \
    if ((expr) != cudaSuccess) {                                                   \
      std::printf("FAIL %s: %s\n", label, cudaGetErrorString(cudaGetLastError())); \
      return 2;                                                                   \
    }                                                                             \
  } while (0)

#define CHECK_EQ(got, want, label)                                     \
  do {                                                                 \
    if ((got) != (want)) {                                             \
      std::printf("FAIL %s: got %lld want %lld\n", label,              \
                  (long long)(got), (long long)(want));                \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

int main() {
  cudaFuncAttributes mid, tiny;
  CHECK_OP(cudaFuncGetAttributes(&mid, (const void*)occmid), "attr mid");
  CHECK_OP(cudaFuncGetAttributes(&tiny, (const void*)occtiny), "attr tiny");
  std::printf("mid: shared=%zu regs=%d local=%zu const=%zu maxT=%d ptx=%d bin=%d dynSh=%d\n",
              mid.sharedSizeBytes, mid.numRegs, mid.localSizeBytes, mid.constSizeBytes,
              mid.maxThreadsPerBlock, mid.ptxVersion, mid.binaryVersion,
              mid.maxDynamicSharedSizeBytes);
  std::printf("tiny: shared=%zu regs=%d local=%zu maxT=%d\n", tiny.sharedSizeBytes, tiny.numRegs,
              tiny.localSizeBytes, tiny.maxThreadsPerBlock);

  for (int bs = 128; bs <= 1024; bs *= 2) {
    int nb = -1;
    CHECK_OP(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, occmid, bs, 0), "occ mid");
    std::printf("mid bs=%d active=%d\n", bs, nb);
  }
  int nb = -1;
  CHECK_OP(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, occmid, 2048, 0), "occ huge");
  std::printf("mid bs=2048 active=%d\n", nb);
  nb = -1;
  CHECK_OP(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, occtiny, 256, 0), "occ tiny");
  std::printf("tiny bs=256 active=%d\n", nb);

  int mg = -1, bs = -1;
  CHECK_OP(cudaOccupancyMaxPotentialBlockSize(&mg, &bs, occmid, 0, 0), "potential");
  std::printf("potential: minGrid=%d block=%d\n", mg, bs);

  size_t avail = 0;
  CHECK_OP(cudaOccupancyAvailableDynamicSMemPerBlock(&avail, occmid, 4, 256), "avail");
  std::printf("availDyn(4,256)=%zu\n", avail);

  if (failures == 0) std::printf("OK\n");
  return failures ? 1 : 0;
}
