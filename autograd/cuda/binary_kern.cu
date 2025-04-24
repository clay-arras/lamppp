#include "binary_kern.cuh"

namespace autograd {

inline namespace cuda {

namespace {

__global__ void equal(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] == B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void not_equal(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] != B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void greater_equal(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] >= B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void less_equal(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] <= B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void greater_than(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] > B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void less_than(int size, const float* A, const float* B, float* C) {
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
  equal<<<blocks, threads>>>(size, d_a, d_b, d_c);

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
  not_equal<<<blocks, threads>>>(size, d_a, d_b, d_c);

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
  greater_equal<<<blocks, threads>>>(size, d_a, d_b, d_c);

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
  less_equal<<<blocks, threads>>>(size, d_a, d_b, d_c);

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
  greater_than<<<blocks, threads>>>(size, d_a, d_b, d_c);

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
  less_than<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

} // namespace cuda

} // namespace autograd