#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "include/lamppp/tensor/cuda/unary_kern.cuh"

namespace lmp::tensor::detail::cuda {

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

// clang-format off
#define INSTANTIATE_UNARY(r, data, elem)                  \
  template void vecExp<elem>(size_t, const elem*, elem*); \
  template void vecLog<elem>(size_t, const elem*, elem*); \
  template void vecRelu<elem>(size_t, const elem*, elem*);

#include "include/lamppp/tensor/supported_types.hpp"
#define TYPES_LIST TYPES()
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_UNARY, , TYPES_LIST)

#undef INSTANTIATE_UNARY
#undef TYPES
// clang-format on

}  // namespace lmp::tensor::detail::cuda
