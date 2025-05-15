#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "lamppp/tensor/cuda/unary_kern.cuh"

namespace lmp::tensor::detail::cuda {

// TODO: gotta figure out a way to do this without having static cast to double in the kernel
template <typename T>
__global__ void vecExpKernel(const T* in, T* out, size_t size) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = exp(static_cast<double>(in[i]));
  }
}

template <typename T>
void vecExp(const T* in, T* out, size_t size) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecExpKernel<<<blocks, threads>>>(in, out, size);
}

template <typename T>
__global__ void vecLogKernel(const T* in, T* out, size_t size) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = log(static_cast<double>(in[i]));
  }
}

template <typename T>
void vecLog(const T* in, T* out, size_t size) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecLogKernel<<<blocks, threads>>>(in, out, size);
}

template <typename T>
__global__ void vecSqrtKernel(const T* in, T* out, size_t size) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = sqrt(static_cast<double>(in[i]));
  }
}

template <typename T>
void vecSqrt(const T* in, T* out, size_t size) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecSqrtKernel<<<blocks, threads>>>(in, out, size);
}

template <typename T>
__global__ void vecAbsKernel(const T* in, T* out, size_t size) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = abs(static_cast<double>(in[i]));
  }
}

template <typename T>
void vecAbs(const T* in, T* out, size_t size) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecAbsKernel<<<blocks, threads>>>(in, out, size);
}

template <typename T>
__global__ void vecSinKernel(const T* in, T* out, size_t size) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = sin(static_cast<double>(in[i]));
  }
}

template <typename T>
void vecSin(const T* in, T* out, size_t size) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecSinKernel<<<blocks, threads>>>(in, out, size);
}

template <typename T>
__global__ void vecCosKernel(const T* in, T* out, size_t size) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = cos(static_cast<double>(in[i]));
  }
}

template <typename T>
void vecCos(const T* in, T* out, size_t size) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecCosKernel<<<blocks, threads>>>(in, out, size);
}

template <typename T>
__global__ void vecTanKernel(const T* in, T* out, size_t size) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = tan(static_cast<double>(in[i]));
  }
}

template <typename T>
void vecTan(const T* in, T* out, size_t size) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecTanKernel<<<blocks, threads>>>(in, out, size);
}

template <typename T>
__global__ void vecClampKernel(const T* in, T min_val, T max_val, T* out,
                               size_t size) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    T val = in[i];
    out[i] = max(min_val, min(val, max_val));
  }
}

template <typename T>
void vecClamp(const T* in, T min_val, T max_val, T* out, size_t size) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecClampKernel<<<blocks, threads>>>(in, min_val, max_val, out, size);
}

// clang-format off
#define INSTANTIATE(r, data, elem)                         \
  template void vecExp<elem>(const elem*, elem*, size_t);  \
  template void vecLog<elem>(const elem*, elem*, size_t);  \
  template void vecSqrt<elem>(const elem*, elem*, size_t); \
  template void vecAbs<elem>(const elem*, elem*, size_t);  \
  template void vecSin<elem>(const elem*, elem*, size_t);  \
  template void vecCos<elem>(const elem*, elem*, size_t);  \
  template void vecTan<elem>(const elem*, elem*, size_t);  \
  template void vecClamp<elem>(const elem*, elem, elem, elem*, size_t);

#include "lamppp/tensor/supported_types.hpp"
#define TYPES_LIST LMP_TYPES()
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, , TYPES_LIST)

#undef INSTANTIATE
// clang-format on

}  // namespace lmp::tensor::detail::cuda
