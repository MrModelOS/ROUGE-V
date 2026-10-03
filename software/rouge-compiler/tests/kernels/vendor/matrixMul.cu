// Vendor kernel in the style of cuda-samples matrixMul (0_MatrixMul),
// BLOCK_SIZE 16 template. Entry: _Z13MatrixMulCUDAILi16EEvPfS0_S0_ii.
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A, float *B,
                                                        int wA, int wB) {
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int aBegin = wA * BLOCK_SIZE * by;
  int aEnd = aBegin + wA - 1;
  int aStep = BLOCK_SIZE;
  int bBegin = BLOCK_SIZE * bx;
  int bStep = BLOCK_SIZE * wB;
  float Csub = 0;
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];
    __syncthreads();
    for (int k = 0; k < BLOCK_SIZE; ++k) Csub += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}
template __global__ void MatrixMulCUDA<16>(float *, float *, float *, int, int);
