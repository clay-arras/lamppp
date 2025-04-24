#include "matrix_kern.cuh"

namespace autograd {

inline namespace cuda {

namespace {

__global__ void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);

    if (i < m && j < n) {
        C[(i*n) + j] = 0;
        for (int t=0; t<k; t++) { // NOTE: A is MxK, B is KxN, C is MxN
            C[(i*n) + j] += A[(i*k) + t] * B[(n*t) + j]; // TODO(nlin): this can be made faster but whatever
        }
    }
}

} // anonymous namespace

extern "C" void cudaMatMul(const float* A, const float* B, float* C, int m, int n, int k) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes_a = m * k * sizeof(float);
  size_t bytes_b = k * n * sizeof(float);
  size_t bytes_c = m * n * sizeof(float);

  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);
  cudaMemcpy(d_a, A, bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes_b, cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((m + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
  matmul<<<blocks, threads>>>(d_a, d_b, d_c, m, n, k);

  cudaMemcpy(C, d_c, bytes_c, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

} // namespace cuda

} // namespace autograd