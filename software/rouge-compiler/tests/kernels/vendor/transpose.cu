// Vendor kernel in the style of cuda-samples transpose (0_MatrixTranspose):
// tiled 32x32 transpose through shared memory with padding against bank
// conflicts. Entry: transposeCoalesced(float*, float*, int).
#define TILE_DIM 32
#define BLOCK_ROWS 8
extern "C" __global__ void transposeCoalesced(float *odata, const float *idata, int width) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
  __syncthreads();
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}
