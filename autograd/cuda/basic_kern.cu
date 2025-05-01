#include "basic_kern.cuh"

namespace autograd {

inline namespace cuda {

template <typename T>
__global__ void vecAddKernel(size_t size, const T* A, const T* B, T* C) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
__global__ void vecSubKernel(size_t size, const T* A, const T* B, T* C) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] - B[i];
    }
}

template <typename T>
__global__ void vecMulKernel(size_t size, const T* A, const T* B, T* C) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] * B[i];
    }
}

template <typename T>
__global__ void vecDivKernel(size_t size, const T* A, const T* B, T* C) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] / B[i];
    }
}

template <typename T>
void vecAdd(size_t size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks  = (size + threads - 1) / threads;
  vecAddKernel<T><<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecSub(size_t size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks  = (size + threads - 1) / threads;
  vecSubKernel<T><<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecMul(size_t size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks  = (size + threads - 1) / threads;
  vecMulKernel<T><<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void vecDiv(size_t size, const T* A, const T* B, T* C) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks  = (size + threads - 1) / threads;
  vecDivKernel<T><<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

#define X(TYPE) template void vecAdd<TYPE>(size_t, const TYPE*, const TYPE*, TYPE*); \
                 template void vecSub<TYPE>(size_t, const TYPE*, const TYPE*, TYPE*); \
                 template void vecMul<TYPE>(size_t, const TYPE*, const TYPE*, TYPE*); \
                 template void vecDiv<TYPE>(size_t, const TYPE*, const TYPE*, TYPE*);
#include "autograd/engine/supported_types.def"
#undef  X


} // namespace cuda
} // namespace autograd