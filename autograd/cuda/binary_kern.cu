#include "binary_kern.cuh"

namespace autograd {

inline namespace cuda {

template <typename T>
__global__ void vecEqualKernel(size_t size, const T* A, const T* B, T* C) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] == B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
__global__ void vecNotEqualKernel(size_t size, const T* A, const T* B, T* C) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] != B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
__global__ void vecGreaterEqualKernel(size_t size, const T* A, const T* B, T* C) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] >= B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
__global__ void vecLessEqualKernel(size_t size, const T* A, const T* B, T* C) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] <= B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
__global__ void vecGreaterThanKernel(size_t size, const T* A, const T* B, T* C) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] > B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
__global__ void vecLessThanKernel(size_t size, const T* A, const T* B, T* C) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] < B[i]) ? 1.0F : 0.0F;
    }
}

template <typename T>
void vecEqual(size_t size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecEqualKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecNotEqual(size_t size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecNotEqualKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecGreaterEqual(size_t size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecGreaterEqualKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecLessEqual(size_t size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecLessEqualKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecGreaterThan(size_t size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecGreaterThanKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecLessThan(size_t size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecLessThanKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

#define X(TYPE) template void vecEqual<TYPE>(size_t, const TYPE*, const TYPE*, TYPE*); \
                 template void vecNotEqual<TYPE>(size_t, const TYPE*, const TYPE*, TYPE*); \
                 template void vecGreaterEqual<TYPE>(size_t, const TYPE*, const TYPE*, TYPE*); \
                 template void vecLessEqual<TYPE>(size_t, const TYPE*, const TYPE*, TYPE*); \
                 template void vecGreaterThan<TYPE>(size_t, const TYPE*, const TYPE*, TYPE*); \
                 template void vecLessThan<TYPE>(size_t, const TYPE*, const TYPE*, TYPE*);
#include "autograd/engine/supported_types.def"
#undef  X

} // namespace cuda

} // namespace autograd