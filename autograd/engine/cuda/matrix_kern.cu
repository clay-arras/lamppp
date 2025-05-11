#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "matrix_kern.cuh"

namespace autograd {

inline namespace cuda {

template <typename U, typename V, typename OutType>
__global__ void cudaMatmulKernel(const U* A, const V* B, OutType* C, size_t m,
                                 size_t n, size_t k) {
  size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
  size_t j = threadIdx.y + (blockIdx.y * blockDim.y);

  if (i < m && j < n) {
    OutType sum = 0;
    for (size_t t = 0; t < k; t++) {
      sum += static_cast<OutType>(A[(i * k) + t]) *
             static_cast<OutType>(B[(n * t) + j]);
    }
    C[(i * n) + j] = sum;
  }
}

template <typename T>
__global__ void cudaTransposeKernel(const T* in, T* out, size_t m, size_t n) {
  size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
  size_t j = threadIdx.y + (blockIdx.y * blockDim.y);

  if (i < m && j < n) {
    out[(j * m) + i] = in[(i * n) + j];
  }
}

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

// clang-format off
#define TYPES (bool)(int)(float)(double)

#define INSTANTIATE_MATMUL(r, product)                                   \
  template void cudaMatMul<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                           BOOST_PP_SEQ_ELEM(1, product), /* V */    \
                           BOOST_PP_SEQ_ELEM(2, product)  /* OutType */    \
                           >(const BOOST_PP_SEQ_ELEM(0, product)*,  \
                             const BOOST_PP_SEQ_ELEM(1, product)*,  \
                             BOOST_PP_SEQ_ELEM(2, product)*, size_t, size_t, size_t);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_MATMUL, (TYPES)(TYPES)(TYPES))

#undef INSTANTIATE_MATMUL

#define X(TYPE) template void cudaTranspose<TYPE>(const TYPE*, TYPE*, size_t, size_t);
#include "autograd/engine/supported_types.def"
#undef X

#undef TYPES
// clang-format on

}  // namespace cuda

}  // namespace autograd