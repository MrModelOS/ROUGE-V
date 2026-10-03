// Vendor kernel in the style of cuda-samples reduction (2_Reduction):
// per-block sum with shared memory, single block-sum written via atomicAdd.
// Entries: blockSum(float*, float*, int), globalSum(float*, int).
extern "C" __global__ void blockSum(const float *g_idata, float *g_odata, int n) {
  __shared__ float sdata[256];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? g_idata[i] : 0;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(g_odata, sdata[0]);
}
extern "C" __global__ void globalSum(float *g_odata, int n) {
  // Second-stage: normalise the accumulated total (exercises atom.global path end-to-end).
  if (threadIdx.x == 0 && blockIdx.x == 0) g_odata[0] = g_odata[0] / (float)n;
}
