// Vendor kernel in the style of cuda-samples threadFenceReduction:
// producer/consumer flag protocol with __threadfence(). Entries:
// fenceProducer(float*, unsigned int*, int), fenceConsumer(float*, unsigned int*, float*, int).
extern "C" __global__ void fenceProducer(float *data, unsigned int *flag, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = (float)i * 1.5f + 0.25f;
    __threadfence();
    atomicExch(flag, 1u);
  }
}
extern "C" __global__ void fenceConsumer(const float *data, const unsigned int *flag,
                                         float *out, int n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    while (*flag == 0) {
    }
    float acc = 0;
    for (int i = 0; i < n; ++i) acc += data[i];
    out[0] = acc;
  }
}
