// Real CUDA C for the vendor-PTX regression test (test_nvcc_ptx).
//
// It is compiled by nvcc and the resulting PTX is fed to ptx2ir. Everything in
// here is deliberately written the way real device code is written, so the
// generated PTX contains the constructs a vendor compiler actually emits:
// one-line .entry signatures, __device__ globals addressed through cvta/mov,
// shared-memory indexing by "reg + constant", __syncthreads, atomics and a
// loop with a predicate-guarded shuffle destination.
__device__ float rouge_global[4096];

__global__ void rouge_real(const float* in, float* out, unsigned n) {
    __shared__ float tile[256];
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = blockDim.x;
    float acc = 0.0f;
    for (unsigned j = i; j < n; j += stride) acc += in[j];

    tile[tid] = acc;
    __syncthreads();
    if (tid < blockDim.x / 2) tile[tid] += tile[tid + blockDim.x / 2];
    __syncthreads();
    if (tid < blockDim.x / 4) tile[tid] += tile[tid + blockDim.x / 4];
    if (tid == 0) out[blockIdx.x] = tile[0];

    atomicAdd(&rouge_global[0], (float)blockIdx.x);
    atomicAdd(&out[blockIdx.x + 64], tile[0]);
}
