#include "binary_kern.cuh"

namespace autograd {

inline namespace cuda {

template <typename T>
__global__ void vecEqualKernel(int size, const T* A, const T* B, T* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] == B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
__global__ void vecNotEqualKernel(int size, const T* A, const T* B, T* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] != B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
__global__ void vecGreaterEqualKernel(int size, const T* A, const T* B, T* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] >= B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
__global__ void vecLessEqualKernel(int size, const T* A, const T* B, T* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] <= B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
__global__ void vecGreaterThanKernel(int size, const T* A, const T* B, T* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] > B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
__global__ void vecLessThanKernel(int size, const T* A, const T* B, T* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] < B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
void vecEqual(int size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

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

template <typename T>
void vecNotEqual(int size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

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

template <typename T>
void vecGreaterEqual(int size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

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

template <typename T>
void vecLessEqual(int size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

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

template <typename T>
void vecGreaterThan(int size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

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

template <typename T>
void vecLessThan(int size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

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

// Explicit template instantiations
template void vecEqual<float>(int size, const float* A, const float* B, float* C);
template void vecNotEqual<float>(int size, const float* A, const float* B, float* C);
template void vecGreaterEqual<float>(int size, const float* A, const float* B, float* C);
template void vecLessEqual<float>(int size, const float* A, const float* B, float* C);
template void vecGreaterThan<float>(int size, const float* A, const float* B, float* C);
template void vecLessThan<float>(int size, const float* A, const float* B, float* C);

} // namespace cuda

} // namespace autograd