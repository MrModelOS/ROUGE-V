// Our own CUDA C kernel: tiled INT8 GEMM with int32 accumulation.
//
// Source of truth for the algorithm: software/rouge-numerics/src/quant_int8.cpp
// (rouge_gemm_i8). The point of this kernel is NOT speed — it is that the
// result is INTEGER, so the GPU answer must match the host answer
// bit-for-bit, with no tolerance at all.
//
// Compiled with the user's own nvcc (13.4) into int8_gemm.ptx; the .ptx then
// goes through OUR ptx2ir and is executed on the card by OUR runtime. No
// NVIDIA library, header or binary is linked into the test: the .ptx is
// loaded as text through the CUDA Driver API, exactly like a third-party
// kernel would be.
//
// Entry: int8_gemm_kernel(const signed char* A, const signed char* B,
//                         int* C, int M, int N, int K)
//   A: [M][K] row-major, B: [K][N] row-major, C: [M][N] int32.

#define TILE 16

extern "C" __global__ void int8_gemm_kernel(const signed char *A,
                                            const signed char *B, int *C, int M,
                                            int N, int K) {
  __shared__ signed char As[TILE][TILE + 1];
  __shared__ signed char Bs[TILE][TILE + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int m = blockIdx.y * TILE + ty;
  const int n = blockIdx.x * TILE + tx;

  int acc = 0;
  const int ktiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < ktiles; ++t) {
    const int a_col = t * TILE + tx;
    const int b_row = t * TILE + ty;
    As[ty][tx] = (m < M && a_col < K) ? A[m * K + a_col] : (signed char)0;
    Bs[ty][tx] = (b_row < K && n < N) ? B[b_row * N + n] : (signed char)0;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k)
      acc += (int)As[ty][k] * (int)Bs[k][tx];
    __syncthreads();
  }
  if (m < M && n < N) C[m * N + n] = acc;
}
