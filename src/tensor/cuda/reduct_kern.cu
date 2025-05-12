#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "include/lamppp/tensor/cuda/reduct_kern.cuh"

namespace lmp::tensor::detail::cuda {

template <typename T>
__global__ void vecSumKernel(const T* in, T* out, const size_t* shape,
                             size_t* stride, size_t axis, size_t outSize) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < outSize) {
    size_t outer = stride[axis + 1];
    size_t inner = stride[axis];
    size_t idx = (i / outer) * inner + (i % outer);

    T sum = 0.0;
    for (size_t j = 0; j < shape[axis]; ++j) {
      sum += in[idx + j * outer];
    }

    out[i] = sum;
  }
}

template <typename T>
__global__ void vecMaxKernel(const T* in, T* out, const size_t* shape,
                             size_t* stride, size_t axis, size_t outSize) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < outSize) {
    size_t outer = stride[axis + 1];
    size_t inner = stride[axis];
    size_t idx = (i / outer) * inner + (i % outer);

    T max = 0.0;
    for (size_t j = 0; j < shape[axis]; ++j) {
      max = fmaxf(max, in[idx + j * outer]);
    }

    out[i] = max;
  }
}

template <typename T>
void vecSum(const T* in, T* out, const size_t* shape, size_t axis,
            size_t ndims) {
  size_t totalSize = 1;
  for (size_t i = 0; i < ndims; ++i) {
    totalSize *= shape[i];
  }
  size_t outSize = totalSize / shape[axis];
  size_t* h_stride = new size_t[ndims + 1];

  h_stride[ndims] = 1;
  for (int i = ndims - 1; i >= 0; i--) {
    h_stride[i] = h_stride[i + 1] * shape[i];
  }

  T *d_in, *d_out;
  size_t *d_shape, *d_stride;

  cudaMalloc(&d_in, totalSize * sizeof(T));
  cudaMalloc(&d_out, outSize * sizeof(T));
  cudaMalloc(&d_shape, ndims * sizeof(size_t));
  cudaMalloc(&d_stride, (ndims + 1) * sizeof(size_t));

  cudaMemcpy(d_in, in, totalSize * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_shape, shape, ndims * sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_stride, h_stride, (ndims + 1) * sizeof(size_t),
             cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (outSize + threads - 1) / threads;
  vecSumKernel<<<blocks, threads>>>(d_in, d_out, d_shape, d_stride, axis,
                                    outSize);

  cudaMemcpy(out, d_out, outSize * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_shape);
  cudaFree(d_stride);

  delete[] h_stride;
}

template <typename T>
void vecMax(const T* in, T* out, const size_t* shape, size_t axis,
            size_t ndims) {
  size_t totalSize = 1;
  for (size_t i = 0; i < ndims; ++i) {
    totalSize *= shape[i];
  }
  size_t outSize = totalSize / shape[axis];
  size_t* h_stride = new size_t[ndims + 1];

  h_stride[ndims] = 1;
  for (int i = ndims - 1; i >= 0; i--) {
    h_stride[i] = h_stride[i + 1] * shape[i];
  }

  T *d_in, *d_out;
  size_t *d_shape, *d_stride;

  cudaMalloc(&d_in, totalSize * sizeof(T));
  cudaMalloc(&d_out, outSize * sizeof(T));
  cudaMalloc(&d_shape, ndims * sizeof(size_t));
  cudaMalloc(&d_stride, (ndims + 1) * sizeof(size_t));

  cudaMemcpy(d_in, in, totalSize * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_shape, shape, ndims * sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_stride, h_stride, (ndims + 1) * sizeof(size_t),
             cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (outSize + threads - 1) / threads;
  vecSumKernel<<<blocks, threads>>>(d_in, d_out, d_shape, d_stride, axis,
                                    outSize);

  cudaMemcpy(out, d_out, outSize * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_shape);
  cudaFree(d_stride);

  delete[] h_stride;
}

// clang-format off
#define INSTANTIATE_REDUCT(r, data, elem) \
  template void vecSum<elem>(const elem*, elem*, const size_t*, size_t, size_t); \
  template void vecMax<elem>(const elem*, elem*, const size_t*, size_t, size_t);

#include "include/lamppp/tensor/supported_types.hpp"
#define TYPES_LIST LMP_TYPES()
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_REDUCT, , TYPES_LIST)

#undef INSTANTIATE_REDUCT
// clang-format on

}  // namespace lmp::tensor::detail::cuda
