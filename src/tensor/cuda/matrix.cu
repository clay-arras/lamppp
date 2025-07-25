#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/cuda/matrix.cuh"
#include "lamppp/tensor/data_type.hpp"

namespace lmp::tensor::detail::cuda {

namespace {
template <typename U, typename V, typename OutType>
__global__ void cudaMatmulKernel(const U* A, const V* B, OutType* C, size_t m,
                                 size_t n, size_t k) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < m;
       i += gridDim.x * blockDim.x) {
    for (size_t j = (blockIdx.y * blockDim.y) + threadIdx.y; j < n;
         j += gridDim.y * blockDim.y) {
      OutType sum = 0;
      // areas of speedup: use reduction strats here
      for (size_t t = 0; t < k; t++) {
        sum += static_cast<OutType>(A[(i * k) + t]) *
               static_cast<OutType>(B[(n * t) + j]);
      }
      C[(i * n) + j] = sum;
    }
  }
}

template <typename T>
__global__ void cudaTransposeKernel(const T* in, T* out, size_t m, size_t n) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < m;
       i += gridDim.x * blockDim.x) {
    for (size_t j = (blockIdx.y * blockDim.y) + threadIdx.y; j < n;
         j += gridDim.y * blockDim.y) {
      out[(j * m) + i] = in[(i * n) + j];
    }
  }
}

}  // namespace

template <typename U, typename V, typename OutType>
void cudaMatMul(const U* A, const V* B, OutType* C, size_t m, size_t n,
                size_t k) {
  dim3 threads(16, 16);
  dim3 blocks((m + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
  cudaMatmulKernel<U, V, OutType><<<blocks, threads>>>(A, B, C, m, n, k);
}

template <typename T>
void cudaTranspose(const T* in, T* out, size_t m, size_t n) {
  dim3 threads(16, 16);
  dim3 blocks((m + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
  cudaTransposeKernel<T><<<blocks, threads>>>(in, out, m, n);
}

#define INSTANTIATE_MATMUL(arg1_type, arg2_type, out_type)  \
  template void cudaMatMul<arg1_type, arg2_type, out_type>( \
      const arg1_type*, const arg2_type*, out_type*, size_t, size_t, size_t);
#define INSTANTIATE_TRANSPOSE(type) \
  template void cudaTranspose<type>(const type*, type*, size_t, size_t);

LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_MATMUL, LMP_LIST_TYPES,
                               LMP_LIST_TYPES, LMP_LIST_TYPES);
LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_TRANSPOSE, LMP_LIST_TYPES);

#undef INSTANTIATE_MATMUL
#undef INSTANTIATE_TRANSPOSE

}  // namespace lmp::tensor::detail::cuda
