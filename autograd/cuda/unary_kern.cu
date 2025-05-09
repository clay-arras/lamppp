#include "unary_kern.cuh"

namespace autograd {

inline namespace cuda {

// TODO: gotta figure out a way to do this without having static cast to double in the kernel
template <typename T>
__global__ void vecExpKernel(size_t size, const T* in, T* out) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = exp(static_cast<double>(in[i]));
  }
}

template <typename T>
__global__ void vecLogKernel(size_t size, const T* in, T* out) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = log(static_cast<double>(in[i]));
  }
}

template <typename T>
__global__ void vecReluKernel(size_t size, const T* in, T* out) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = in[i] > 0 ? in[i] : 0;
  }
}

template <typename T>
void vecExp(size_t size, const T* in, T* out) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecExpKernel<<<blocks, threads>>>(size, in, out);
}

template <typename T>
void vecLog(size_t size, const T* in, T* out) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecLogKernel<<<blocks, threads>>>(size, in, out);
}

template <typename T>
void vecRelu(size_t size, const T* in, T* out) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecReluKernel<<<blocks, threads>>>(size, in, out);
}

#define X(TYPE)                                           \
  template void vecExp<TYPE>(size_t, const TYPE*, TYPE*); \
  template void vecLog<TYPE>(size_t, const TYPE*, TYPE*); \
  template void vecRelu<TYPE>(size_t, const TYPE*, TYPE*);
#include "autograd/engine/supported_types.def"
#undef X

}  // namespace cuda

}  // namespace autograd