#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "binary_kern.cuh"

namespace autograd {

inline namespace cuda {

template <typename U, typename V>
__global__ void vecEqualKernel(size_t size, const U* A, const V* B, bool* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = (A[i] == B[i]);
  }
}

template <typename U, typename V>
__global__ void vecNotEqualKernel(size_t size, const U* A, const V* B,
                                  bool* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = (A[i] != B[i]);
  }
}

template <typename U, typename V>
__global__ void vecGreaterEqualKernel(size_t size, const U* A, const V* B,
                                      bool* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = (A[i] >= B[i]);
  }
}

template <typename U, typename V>
__global__ void vecLessEqualKernel(size_t size, const U* A, const V* B,
                                   bool* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = (A[i] <= B[i]);
  }
}

template <typename U, typename V>
__global__ void vecGreaterThanKernel(size_t size, const U* A, const V* B,
                                     bool* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = (A[i] > B[i]);
  }
}

template <typename U, typename V>
__global__ void vecLessThanKernel(size_t size, const U* A, const V* B,
                                  bool* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = (A[i] < B[i]);
  }
}

template <typename U, typename V>
void vecEqual(size_t size, const U* A, const V* B, bool* C) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V>
void vecNotEqual(size_t size, const U* A, const V* B, bool* C) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecNotEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V>
void vecGreaterEqual(size_t size, const U* A, const V* B, bool* C) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecGreaterEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V>
void vecLessEqual(size_t size, const U* A, const V* B, bool* C) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecLessEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V>
void vecGreaterThan(size_t size, const U* A, const V* B, bool* C) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecGreaterThanKernel<U, V><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V>
void vecLessThan(size_t size, const U* A, const V* B, bool* C) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecLessThanKernel<U, V><<<blocks, threads>>>(size, A, B, C);
}

// clang-format off
#define TYPES (bool)(int)(float)(double)

#define INSTANTIATE_COMPARISON(r, product)                                   \
  template void vecEqual<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                         BOOST_PP_SEQ_ELEM(1, product)  /* V */    \
                         >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,  \
                                   const BOOST_PP_SEQ_ELEM(1, product)*,  \
                                   bool*); \
  template void vecNotEqual<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                           BOOST_PP_SEQ_ELEM(1, product)  /* V */    \
                           >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,  \
                                     const BOOST_PP_SEQ_ELEM(1, product)*,  \
                                     bool*); \
  template void vecGreaterEqual<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                               BOOST_PP_SEQ_ELEM(1, product)  /* V */    \
                               >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,  \
                                         const BOOST_PP_SEQ_ELEM(1, product)*,  \
                                         bool*); \
  template void vecLessEqual<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                            BOOST_PP_SEQ_ELEM(1, product)  /* V */    \
                            >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,  \
                                      const BOOST_PP_SEQ_ELEM(1, product)*,  \
                                      bool*); \
  template void vecGreaterThan<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                              BOOST_PP_SEQ_ELEM(1, product)  /* V */    \
                              >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,  \
                                        const BOOST_PP_SEQ_ELEM(1, product)*,  \
                                        bool*); \
  template void vecLessThan<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                           BOOST_PP_SEQ_ELEM(1, product)  /* V */    \
                           >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,  \
                                     const BOOST_PP_SEQ_ELEM(1, product)*,  \
                                     bool*);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_COMPARISON, (TYPES)(TYPES))

#undef INSTANTIATE_COMPARISON
#undef TYPES
// clang-format on

}  // namespace cuda

}  // namespace autograd