#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <cassert>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/cuda/reduct_kern.cuh"

namespace lmp::tensor::detail::cuda {

template <typename T>
__global__ void vecSumKernel(const T* in, T* out, const size_t* shape,
                             stride_t* stride, size_t axis, size_t outSize) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < outSize) {
    stride_t outer = stride[axis];
    stride_t inner = stride[axis - 1];
    stride_t idx = (i / outer) * inner + (i % outer);

    T sum = 0.0;
    for (size_t j = 0; j < shape[axis]; ++j) {
      sum += in[idx + j * outer];
    }

    out[i] = sum;
  }
}

template <typename T>
__global__ void vecMaxKernel(const T* in, T* out, const size_t* shape,
                             stride_t* stride, size_t axis, size_t outSize) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < outSize) {
    stride_t outer = stride[axis];
    stride_t inner = stride[axis - 1];
    stride_t idx = (i / outer) * inner + (i % outer);

    T val = in[idx];
    for (size_t j = 1; j < shape[axis]; ++j) {
      T current_val = in[idx + j * outer];
      if (current_val > val) {
        val = current_val;
      }
    }
    out[i] = val;
  }
}

template <typename T>
__global__ void vecMinKernel(const T* in, T* out, const size_t* shape,
                             stride_t* stride, size_t axis, size_t outSize) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < outSize) {
    stride_t outer = stride[axis];
    stride_t inner = stride[axis - 1];
    stride_t idx = (i / outer) * inner + (i % outer);

    T val = in[idx];
    for (size_t j = 1; j < shape[axis]; ++j) {
      T current_val = in[idx + j * outer];
      if (current_val < val) {
        val = current_val;
      }
    }
    out[i] = val;
  }
}

template <typename T>
void vecSum(const T* in, T* out, const size_t* shape, const stride_t* stride,
            size_t axis, size_t ndims, size_t size) {

  size_t* d_shape;
  stride_t* d_stride;
  cudaMalloc(&d_shape, ndims * sizeof(size_t));
  cudaMalloc(&d_stride, ndims * sizeof(stride_t));

  cudaError_t err;
  err = cudaMemcpy(d_shape, shape, ndims * sizeof(size_t),
                   cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  err = cudaMemcpy(d_stride, stride, ndims * sizeof(stride_t),
                   cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  size_t outSize = size / shape[axis];
  size_t threads = 256;
  size_t blocks = (outSize + threads - 1) / threads;
  vecSumKernel<<<blocks, threads>>>(const_cast<T*>(in), out, d_shape, d_stride,
                                    axis, outSize);

  cudaFree(d_shape);
  cudaFree(d_stride);
}

template <typename T>
void vecMax(const T* in, T* out, const size_t* shape, const stride_t* stride,
            size_t axis, size_t ndims, size_t size) {
  size_t* d_shape;
  stride_t* d_stride;
  cudaMalloc(&d_shape, ndims * sizeof(size_t));
  cudaMalloc(&d_stride, ndims * sizeof(stride_t));

  cudaError_t err;
  err = cudaMemcpy(d_shape, shape, ndims * sizeof(size_t),
                   cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  err = cudaMemcpy(d_stride, stride, ndims * sizeof(stride_t),
                   cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  size_t outSize = size / shape[axis];
  size_t threads = 256;
  size_t blocks = (outSize + threads - 1) / threads;
  vecMaxKernel<<<blocks, threads>>>(const_cast<T*>(in), out, d_shape, d_stride,
                                    axis, outSize);

  cudaFree(d_shape);
  cudaFree(d_stride);
}

template <typename T>
void vecMin(const T* in, T* out, const size_t* shape, const stride_t* stride,
            size_t axis, size_t ndims, size_t size) {
  size_t* d_shape;
  stride_t* d_stride;
  cudaMalloc(&d_shape, ndims * sizeof(size_t));
  cudaMalloc(&d_stride, ndims * sizeof(stride_t));

  cudaError_t err;
  err = cudaMemcpy(d_shape, shape, ndims * sizeof(size_t),
                   cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  err = cudaMemcpy(d_stride, stride, ndims * sizeof(stride_t),
                   cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  size_t outSize = size / shape[axis];
  size_t threads = 256;
  size_t blocks = (outSize + threads - 1) / threads;
  vecMinKernel<<<blocks, threads>>>(const_cast<T*>(in), out, d_shape, d_stride,
                                    axis, outSize);

  cudaFree(d_shape);
  cudaFree(d_stride);
}

// clang-format off
#define INSTANTIATE_REDUCT(r, data, elem) \
  template void vecSum<elem>(const elem*, elem*, const size_t*, const stride_t*, size_t, size_t, size_t); \
  template void vecMax<elem>(const elem*, elem*, const size_t*, const stride_t*, size_t, size_t, size_t); \
  template void vecMin<elem>(const elem*, elem*, const size_t*, const stride_t*, size_t, size_t, size_t); 

#include "lamppp/tensor/supported_types.hpp"
#define TYPES_LIST LMP_TYPES()
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_REDUCT, , TYPES_LIST)

#undef INSTANTIATE_REDUCT
// clang-format on

}  // namespace lmp::tensor::detail::cuda
