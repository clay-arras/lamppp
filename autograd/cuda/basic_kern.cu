#include "basic_kern.cuh"

namespace autograd {

inline namespace cuda {

// Kernels moved outside anonymous namespace
template <typename T>
__global__ void vecAddKernel(int size, const T* A, const T* B, T* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
__global__ void vecSubKernel(int size, const T* A, const T* B, T* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] - B[i];
    }
}

template <typename T>
__global__ void vecMulKernel(int size, const T* A, const T* B, T* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] * B[i];
    }
}

template <typename T>
__global__ void vecDivKernel(int size, const T* A, const T* B, T* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] / B[i];
    }
}

template <typename T>
void vecAdd(int size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks  = (size + threads - 1) / threads;
  vecAddKernel<T><<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecSub(int size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks  = (size + threads - 1) / threads;
  vecSubKernel<T><<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecMul(int size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks  = (size + threads - 1) / threads;
  vecMulKernel<T><<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecDiv(int size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks  = (size + threads - 1) / threads;
  vecDivKernel<T><<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}


// Explicit template instantiations
template void vecAdd<float>(int size, const float* A, const float* B, float* C);
template void vecSub<float>(int size, const float* A, const float* B, float* C);
template void vecMul<float>(int size, const float* A, const float* B, float* C);
template void vecDiv<float>(int size, const float* A, const float* B, float* C);

} // namespace cuda
} // namespace autograd