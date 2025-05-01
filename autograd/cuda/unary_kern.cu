#include "unary_kern.cuh"

namespace autograd {

inline namespace cuda {

template <typename T>
__global__ void vecExpKernel(size_t size, T* in, T* out) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = expf(in[i]);
    }
}

template <typename T>
__global__ void vecLogKernel(size_t size, T* in, T* out) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = logf(in[i]);
    }
}

template <typename T>
__global__ void vecReluKernel(size_t size, T* in, T* out) {
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = fmaxf(0.0F, in[i]);
    }
}

template <typename T>
void vecExp(size_t size, const T* in, T* out) {
  T *d_in;
  T *d_out;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecExpKernel<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

template <typename T>
void vecLog(size_t size, const T* in, T* out) {
  T *d_in;
  T *d_out;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecLogKernel<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

template <typename T>
void vecRelu(size_t size, const T* in, T* out) {
  T *d_in;
  T *d_out;
  size_t bytes = size * sizeof(T);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecReluKernel<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

#define X(TYPE) template void vecExp<TYPE>(size_t, const TYPE*, TYPE*); \
                 template void vecLog<TYPE>(size_t, const TYPE*, TYPE*); \
                 template void vecRelu<TYPE>(size_t, const TYPE*, TYPE*);
#include "autograd/engine/supported_types.def"
#undef  X

} // namespace cuda

} // namespace autograd