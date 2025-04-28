#include "unary_kern.cuh"

namespace autograd {

inline namespace cuda {

// Kernels moved outside anonymous namespace
template <typename T>
__global__ void vecExpKernel(int size, T* in, T* out) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = expf(in[i]);
    }
}

template <typename T>
__global__ void vecLogKernel(int size, T* in, T* out) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = logf(in[i]);
    }
}

template <typename T>
__global__ void vecReluKernel(int size, T* in, T* out) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = fmaxf(0.0F, in[i]);
    }
}

template <typename T>
void vecExp(int size, const T* in, T* out) {
  T *d_in;
  T *d_out;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecExpKernel<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

template <typename T>
void vecLog(int size, const T* in, T* out) {
  T *d_in;
  T *d_out;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecLogKernel<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

template <typename T>
void vecRelu(int size, const T* in, T* out) {
  T *d_in;
  T *d_out;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecReluKernel<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

// Explicit template instantiations
template void vecExp<float>(int size, const float* in, float* out);
template void vecLog<float>(int size, const float* in, float* out);
template void vecRelu<float>(int size, const float* in, float* out);

} // namespace cuda

} // namespace autograd