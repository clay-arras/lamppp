#include <linux/limits.h>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "lamppp/tensor/cuda/basic_kern.cuh"

namespace lmp::tensor::detail::cuda {

template <typename U, typename V, typename OutType>
__global__ void vecAddKernel(size_t size, const U* A, const V* B, OutType* C,
                             const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array<stride_t, NVARS> offsets = meta->get(i);
    C[offsets[2]] = static_cast<OutType>(A[offsets[0]]) +
                    static_cast<OutType>(B[offsets[1]]);
  }
}

template <typename U, typename V, typename OutType>
__global__ void vecSubKernel(size_t size, const U* A, const V* B, OutType* C,
                             const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array<stride_t, NVARS> offsets = meta->get(i);
    C[offsets[2]] = static_cast<OutType>(A[offsets[0]]) -
                    static_cast<OutType>(B[offsets[1]]);
  }
}

template <typename U, typename V, typename OutType>
__global__ void vecMulKernel(size_t size, const U* A, const V* B, OutType* C,
                             const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array<stride_t, NVARS> offsets = meta->get(i);
    C[offsets[2]] = static_cast<OutType>(A[offsets[0]]) *
                    static_cast<OutType>(B[offsets[1]]);
  }
}

template <typename U, typename V, typename OutType>
__global__ void vecDivKernel(size_t size, const U* A, const V* B, OutType* C,
                             const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array<stride_t, NVARS> offsets = meta->get(i);
    C[offsets[2]] = static_cast<OutType>(A[offsets[0]]) /
                    static_cast<OutType>(B[offsets[1]]);
  }
}

template <typename U, typename V, typename OutType>
void vecAdd(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  OffsetUtil* d_meta;
  cudaError_t err = cudaMalloc(&d_meta, sizeof(OffsetUtil));
  assert(err == cudaSuccess &&
         "vecAdd: Failed to allocate device memory for meta");

  err = cudaMemcpy(d_meta, meta, sizeof(OffsetUtil), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "vecAdd: Failed to copy meta to device");

  vecAddKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C, d_meta);

  cudaFree(d_meta);
}

template <typename U, typename V, typename OutType>
void vecSub(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  OffsetUtil* d_meta;
  cudaError_t err = cudaMalloc(&d_meta, sizeof(OffsetUtil));
  assert(err == cudaSuccess &&
         "vecSub: Failed to allocate device memory for meta");

  err = cudaMemcpy(d_meta, meta, sizeof(OffsetUtil), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "vecSub: Failed to copy meta to device");

  vecSubKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C, d_meta);

  cudaFree(d_meta);
}

template <typename U, typename V, typename OutType>
void vecMul(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  OffsetUtil* d_meta;
  cudaError_t err = cudaMalloc(&d_meta, sizeof(OffsetUtil));
  assert(err == cudaSuccess &&
         "vecMul: Failed to allocate device memory for meta");

  err = cudaMemcpy(d_meta, meta, sizeof(OffsetUtil), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "vecMul: Failed to copy meta to device");

  vecMulKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C, d_meta);

  cudaFree(d_meta);
}

template <typename U, typename V, typename OutType>
void vecDiv(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  OffsetUtil* d_meta;
  cudaError_t err = cudaMalloc(&d_meta, sizeof(OffsetUtil));
  assert(err == cudaSuccess &&
         "vecDiv: Failed to allocate device memory for meta");

  err = cudaMemcpy(d_meta, meta, sizeof(OffsetUtil), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "vecDiv: Failed to copy meta to device");

  vecDivKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C, d_meta);

  cudaFree(d_meta);
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
                         BOOST_PP_SEQ_ELEM(2, product)*, const OffsetUtil*); \
  template void vecMul<BOOST_PP_SEQ_ELEM(0, product), /* U */               \
                       BOOST_PP_SEQ_ELEM(1, product), /* V */               \
                       BOOST_PP_SEQ_ELEM(2, product)  /* T */               \
                       >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,      \
                         const BOOST_PP_SEQ_ELEM(1, product)*,              \
                         BOOST_PP_SEQ_ELEM(2, product)*, const OffsetUtil*); \
  template void vecDiv<BOOST_PP_SEQ_ELEM(0, product), /* U */               \
                       BOOST_PP_SEQ_ELEM(1, product), /* V */               \
                       BOOST_PP_SEQ_ELEM(2, product)  /* T */               \
                       >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,      \
                         const BOOST_PP_SEQ_ELEM(1, product)*,              \
                         BOOST_PP_SEQ_ELEM(2, product)*, const OffsetUtil*);

#include "lamppp/tensor/supported_types.hpp"
#define TYPES_LIST LMP_TYPES()
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (TYPES_LIST)(TYPES_LIST)(TYPES_LIST))

#undef INSTANTIATE
// clang-format on

}  // namespace lmp::tensor::detail::cuda