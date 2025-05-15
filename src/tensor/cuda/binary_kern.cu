#include <boost/preprocessor/control/iif.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "include/lamppp/tensor/cuda/binary_kern.cuh"
#include "include/lamppp/tensor/cuda/offset_util.cuh"

namespace lmp::tensor::detail::cuda {

template <typename U, typename V>
__global__ void vecEqualKernel(size_t size, const U* A, const V* B, bool* C,
                               const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array<stride_t, NVARS> offsets = meta->get(i);
    C[offsets[2]] = (A[offsets[0]] == B[offsets[1]]);
  }
}

template <typename U, typename V>
__global__ void vecNotEqualKernel(size_t size, const U* A, const V* B, bool* C,
                                  const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array<stride_t, NVARS> offsets = meta->get(i);
    C[offsets[2]] = (A[offsets[0]] != B[offsets[1]]);
  }
}

template <typename U, typename V>
__global__ void vecGreaterEqualKernel(size_t size, const U* A, const V* B,
                                      bool* C, const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array<stride_t, NVARS> offsets = meta->get(i);
    C[offsets[2]] = (A[offsets[0]] >= B[offsets[1]]);
  }
}

template <typename U, typename V>
__global__ void vecLessEqualKernel(size_t size, const U* A, const V* B, bool* C,
                                   const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array<stride_t, NVARS> offsets = meta->get(i);
    C[offsets[2]] = (A[offsets[0]] <= B[offsets[1]]);
  }
}

template <typename U, typename V>
__global__ void vecGreaterThanKernel(size_t size, const U* A, const V* B,
                                     bool* C, const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array<stride_t, NVARS> offsets = meta->get(i);
    C[offsets[2]] = (A[offsets[0]] > B[offsets[1]]);
  }
}

template <typename U, typename V>
__global__ void vecLessThanKernel(size_t size, const U* A, const V* B, bool* C,
                                  const OffsetUtil* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array<stride_t, NVARS> offsets = meta->get(i);
    C[offsets[2]] = (A[offsets[0]] < B[offsets[1]]);
  }
}

template <typename U, typename V>
void vecEqual(size_t size, const U* A, const V* B, bool* C,
              const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  OffsetUtil* d_meta;
  cudaError_t err = cudaMalloc(&d_meta, sizeof(OffsetUtil));
  assert(err == cudaSuccess &&
         "vecEqual: Failed to allocate device memory for meta");
  err = cudaMemcpy(d_meta, meta, sizeof(OffsetUtil), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "vecEqual: Failed to copy meta to device");
  vecEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta);
  cudaFree(d_meta);
}

template <typename U, typename V>
void vecNotEqual(size_t size, const U* A, const V* B, bool* C,
                 const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  OffsetUtil* d_meta;
  cudaMalloc(&d_meta, sizeof(OffsetUtil));
  cudaError_t err =
      cudaMemcpy(d_meta, meta, sizeof(OffsetUtil), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "vecNotEqual: Failed to copy meta to device");
  vecNotEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta);
  cudaFree(d_meta);
}

template <typename U, typename V>
void vecGreaterEqual(size_t size, const U* A, const V* B, bool* C,
                     const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  OffsetUtil* d_meta;
  cudaMalloc(&d_meta, sizeof(OffsetUtil));
  cudaError_t err =
      cudaMemcpy(d_meta, meta, sizeof(OffsetUtil), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess &&
         "vecGreaterEqual: Failed to copy meta to device");
  vecGreaterEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta);
  cudaFree(d_meta);
}

template <typename U, typename V>
void vecLessEqual(size_t size, const U* A, const V* B, bool* C,
                  const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  OffsetUtil* d_meta;
  cudaMalloc(&d_meta, sizeof(OffsetUtil));
  cudaError_t err =
      cudaMemcpy(d_meta, meta, sizeof(OffsetUtil), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "vecLessEqual: Failed to copy meta to device");
  vecLessEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta);
  cudaFree(d_meta);
}

template <typename U, typename V>
void vecGreaterThan(size_t size, const U* A, const V* B, bool* C,
                    const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  OffsetUtil* d_meta;
  cudaMalloc(&d_meta, sizeof(OffsetUtil));
  cudaError_t err =
      cudaMemcpy(d_meta, meta, sizeof(OffsetUtil), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "vecGreaterThan: Failed to copy meta to device");
  vecGreaterThanKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta);
  cudaFree(d_meta);
}

template <typename U, typename V>
void vecLessThan(size_t size, const U* A, const V* B, bool* C,
                 const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  OffsetUtil* d_meta;
  cudaMalloc(&d_meta, sizeof(OffsetUtil));
  cudaError_t err =
      cudaMemcpy(d_meta, meta, sizeof(OffsetUtil), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "vecLessThan: Failed to copy meta to device");
  vecLessThanKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta);
  cudaFree(d_meta);
}

// clang-format off
#define INSTANTIATE_COMPARISON(r, product)                                     \
  template void vecEqual<BOOST_PP_SEQ_ELEM(0, product), /* U */                \
                         BOOST_PP_SEQ_ELEM(1, product)  /* V */                \
                         >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,       \
                           const BOOST_PP_SEQ_ELEM(1, product)*, bool*,        \
                           const OffsetUtil*);                                 \
  template void vecNotEqual<BOOST_PP_SEQ_ELEM(0, product), /* U */             \
                            BOOST_PP_SEQ_ELEM(1, product)  /* V */             \
                            >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,    \
                              const BOOST_PP_SEQ_ELEM(1, product)*, bool*,     \
                              const OffsetUtil*);                               \
  template void vecGreaterEqual<BOOST_PP_SEQ_ELEM(0, product), /* U */         \
                                BOOST_PP_SEQ_ELEM(1, product)  /* V */         \
                                >(                                             \
      size_t, const BOOST_PP_SEQ_ELEM(0, product)*,                            \
      const BOOST_PP_SEQ_ELEM(1, product)*, bool*, const OffsetUtil*);         \
  template void vecLessEqual<BOOST_PP_SEQ_ELEM(0, product), /* U */            \
                             BOOST_PP_SEQ_ELEM(1, product)  /* V */            \
                             >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,   \
                               const BOOST_PP_SEQ_ELEM(1, product)*, bool*,    \
                               const OffsetUtil*);                              \
  template void vecGreaterThan<BOOST_PP_SEQ_ELEM(0, product), /* U */          \
                               BOOST_PP_SEQ_ELEM(1, product)  /* V */          \
                               >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*, \
                                 const BOOST_PP_SEQ_ELEM(1, product)*, bool*,  \
                                 const OffsetUtil*);                            \
  template void vecLessThan<BOOST_PP_SEQ_ELEM(0, product), /* U */             \
                            BOOST_PP_SEQ_ELEM(1, product)  /* V */             \
                            >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*,    \
                              const BOOST_PP_SEQ_ELEM(1, product)*, bool*,     \
                              const OffsetUtil*);

#include "include/lamppp/tensor/supported_types.hpp"
#define TYPES_LIST LMP_TYPES()
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_COMPARISON, (TYPES_LIST)(TYPES_LIST))

#undef INSTANTIATE_COMPARISON
// clang-format on

}  // namespace lmp::tensor::detail::cuda
