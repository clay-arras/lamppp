#include <linux/limits.h>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "include/lamppp/tensor/cuda/basic_kern.cuh"
#include "include/lamppp/tensor/data_type.hpp"

namespace lmp::tensor::detail::cuda {

template <typename U, typename V, typename OutType>
__global__ void vecAddKernel(size_t size, const U* A, const V* B, OutType* C,
                             const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    internal::OffsetPair offsets = meta->get(i);
    size_t idxA = offsets.offset1;
    size_t idxB = offsets.offset1;
    C[i] = static_cast<OutType>(A[idxA]) + static_cast<OutType>(B[idxB]);
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
void vecAdd(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecAddKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C, meta);
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
#define INSTANTIATE(r, product)                                             \
  template void vecAdd<BOOST_PP_SEQ_ELEM(0, product), /* U */               \
                       BOOST_PP_SEQ_ELEM(1, product), /* V */               \
                       BOOST_PP_SEQ_ELEM(2, product)  /* T */               \
                       >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,      \
                         const BOOST_PP_SEQ_ELEM(1, product)*,              \
                         BOOST_PP_SEQ_ELEM(2, product)*, const OffsetUtil*); \
  template void vecSub<BOOST_PP_SEQ_ELEM(0, product), /* U */               \
                       BOOST_PP_SEQ_ELEM(1, product), /* V */               \
                       BOOST_PP_SEQ_ELEM(2, product)  /* T */               \
                       >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,      \
                         const BOOST_PP_SEQ_ELEM(1, product)*,              \
                         BOOST_PP_SEQ_ELEM(2, product)*);                   \
  template void vecMul<BOOST_PP_SEQ_ELEM(0, product), /* U */               \
                       BOOST_PP_SEQ_ELEM(1, product), /* V */               \
                       BOOST_PP_SEQ_ELEM(2, product)  /* T */               \
                       >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,      \
                         const BOOST_PP_SEQ_ELEM(1, product)*,              \
                         BOOST_PP_SEQ_ELEM(2, product)*);                   \
  template void vecDiv<BOOST_PP_SEQ_ELEM(0, product), /* U */               \
                       BOOST_PP_SEQ_ELEM(1, product), /* V */               \
                       BOOST_PP_SEQ_ELEM(2, product)  /* T */               \
                       >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,      \
                         const BOOST_PP_SEQ_ELEM(1, product)*,              \
                         BOOST_PP_SEQ_ELEM(2, product)*);

#include "include/lamppp/tensor/supported_types.hpp"
#define TYPES_LIST LMP_TYPES()
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (TYPES_LIST)(TYPES_LIST)(TYPES_LIST))

#undef INSTANTIATE
// clang-format on

}  // namespace lmp::tensor::detail::cuda