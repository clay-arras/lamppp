#include <linux/limits.h>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "basic_kern.cuh"

namespace autograd {

inline namespace cuda {

template <typename U, typename V, typename OutType>
__global__ void vecAddKernel(size_t size, const U* A, const V* B, OutType* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = static_cast<OutType>(A[i]) + static_cast<OutType>(B[i]);
  }
}

template <typename U, typename V, typename OutType>
__global__ void vecSubKernel(size_t size, const U* A, const V* B, OutType* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = static_cast<OutType>(A[i]) - static_cast<OutType>(B[i]);
  }
}

template <typename U, typename V, typename OutType>
__global__ void vecMulKernel(size_t size, const U* A, const V* B, OutType* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = static_cast<OutType>(A[i]) * static_cast<OutType>(B[i]);
  }
}

template <typename U, typename V, typename OutType>
__global__ void vecDivKernel(size_t size, const U* A, const V* B, OutType* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = static_cast<OutType>(A[i]) / static_cast<OutType>(B[i]);
  }
}

template <typename U, typename V, typename OutType>
void vecAdd(size_t size, const U* A, const V* B, OutType* C) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecAddKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V, typename OutType>
void vecSub(size_t size, const U* A, const V* B, OutType* C) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecSubKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V, typename OutType>
void vecMul(size_t size, const U* A, const V* B, OutType* C) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecMulKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V, typename OutType>
void vecDiv(size_t size, const U* A, const V* B, OutType* C) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecDivKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C);
}

// clang-format off
// TODO: make this fill from supported_types.def
#define TYPES (bool)(int)(float)(double)

#define INSTANTIATE(r, product)                                   \
  template void vecAdd<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                      BOOST_PP_SEQ_ELEM(1, product), /* V */    \
                      BOOST_PP_SEQ_ELEM(2, product)  /* T */    \
                      >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,  \
                                const BOOST_PP_SEQ_ELEM(1, product)*,  \
                                BOOST_PP_SEQ_ELEM(2, product)*); \
  template void vecSub<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                      BOOST_PP_SEQ_ELEM(1, product), /* V */    \
                      BOOST_PP_SEQ_ELEM(2, product)  /* T */    \
                      >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,  \
                                const BOOST_PP_SEQ_ELEM(1, product)*,  \
                                BOOST_PP_SEQ_ELEM(2, product)*);  \
  template void vecMul<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                      BOOST_PP_SEQ_ELEM(1, product), /* V */    \
                      BOOST_PP_SEQ_ELEM(2, product)  /* T */    \
                      >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,  \
                                const BOOST_PP_SEQ_ELEM(1, product)*,  \
                                BOOST_PP_SEQ_ELEM(2, product)*); \
  template void vecDiv<BOOST_PP_SEQ_ELEM(0, product), /* U */    \
                      BOOST_PP_SEQ_ELEM(1, product), /* V */    \
                      BOOST_PP_SEQ_ELEM(2, product)  /* T */    \
                      >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,  \
                                const BOOST_PP_SEQ_ELEM(1, product)*,  \
                                BOOST_PP_SEQ_ELEM(2, product)*); 


BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (TYPES)(TYPES)(TYPES))

#undef INSTANTIATE
#undef TYPES
// clang-format on

}  // namespace cuda
}  // namespace autograd