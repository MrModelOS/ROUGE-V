// Vendor kernel in the style of cuda-samples vectorAdd (0_VectorAdd).
// Compiled with the user's nvcc to vendor/vecAdd.ptx; the .ptx is genuine
// nvcc output, never hand-written. Entry: VecAdd_kernel(float*,float*,float*,int).
extern "C" __global__ void VecAdd_kernel(const float *A, const float *B, float *C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) C[i] = A[i] + B[i];
}
