#include "binary_kern.cuh"

namespace autograd {

inline namespace cuda {

namespace {

__global__ void vecEqualKernel(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] == B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void vecNotEqualKernel(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] != B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void vecGreaterEqualKernel(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] >= B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void vecLessEqualKernel(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] <= B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void vecGreaterThanKernel(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] > B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void vecLessThanKernel(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] < B[i]) ? 1.0F : 0.0F;
    }
}

} // anonymous namespace

extern "C" void vecEqual(int size, const float* A, const float* B, float* C) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecEqualKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

extern "C" void vecNotEqual(int size, const float* A, const float* B, float* C) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecNotEqualKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

extern "C" void vecGreaterEqual(int size, const float* A, const float* B, float* C) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecGreaterEqualKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

extern "C" void vecLessEqual(int size, const float* A, const float* B, float* C) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecLessEqualKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

extern "C" void vecGreaterThan(int size, const float* A, const float* B, float* C) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecGreaterThanKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

extern "C" void vecLessThan(int size, const float* A, const float* B, float* C) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecLessThanKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

} // namespace cuda

} // namespace autograd