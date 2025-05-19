#include <boost/preprocessor/control/iif.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "lamppp/tensor/cuda/binary_kern.cuh"
#include "lamppp/tensor/cuda/list_ptr.cuh"
#include "lamppp/tensor/cuda/offset_util.cuh"

namespace lmp::tensor::detail::cuda {

template <typename U, typename V>
__global__ void vecEqualKernel(size_t size, const U* A, const V* B, bool* C,
                               const OffsetUtil<2>* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array offsets = meta->get(i);
    C[offsets[0]] = (static_cast<V>(A[offsets[1]]) == B[offsets[2]]);
  }
}

template <typename U, typename V>
__global__ void vecNotEqualKernel(size_t size, const U* A, const V* B, bool* C,
                                  const OffsetUtil<2>* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array offsets = meta->get(i);
    C[offsets[0]] = (static_cast<V>(A[offsets[1]]) != B[offsets[2]]);
  }
}

template <typename U, typename V>
__global__ void vecGreaterEqualKernel(size_t size, const U* A, const V* B,
                                      bool* C, const OffsetUtil<2>* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array offsets = meta->get(i);
    C[offsets[0]] = (static_cast<V>(A[offsets[1]]) >= B[offsets[2]]);
  }
}

template <typename U, typename V>
__global__ void vecLessEqualKernel(size_t size, const U* A, const V* B, bool* C,
                                   const OffsetUtil<2>* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array offsets = meta->get(i);
    C[offsets[0]] = (static_cast<V>(A[offsets[1]]) <= B[offsets[2]]);
  }
}

template <typename U, typename V>
__global__ void vecGreaterThanKernel(size_t size, const U* A, const V* B,
                                     bool* C, const OffsetUtil<2>* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array offsets = meta->get(i);
    C[offsets[0]] = (static_cast<V>(A[offsets[1]]) > B[offsets[2]]);
  }
}

template <typename U, typename V>
__global__ void vecLessThanKernel(size_t size, const U* A, const V* B, bool* C,
                                  const OffsetUtil<2>* meta) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array offsets = meta->get(i);
    C[offsets[0]] = (static_cast<V>(A[offsets[1]]) < B[offsets[2]]);
  }
}

template <typename U, typename V>
void vecEqual(size_t size, const U* A, const V* B, bool* C,
              const OffsetUtil<2>* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  ListDevicePtr<OffsetUtil<2>> d_meta(meta, 1);
  vecEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta.get());
}

template <typename U, typename V>
void vecNotEqual(size_t size, const U* A, const V* B, bool* C,
                 const OffsetUtil<2>* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  ListDevicePtr<OffsetUtil<2>> d_meta(meta, 1);
  vecNotEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta.get());
}

template <typename U, typename V>
void vecGreaterEqual(size_t size, const U* A, const V* B, bool* C,
                     const OffsetUtil<2>* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  ListDevicePtr<OffsetUtil<2>> d_meta(meta, 1);
  vecGreaterEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta.get());
}

template <typename U, typename V>
void vecLessEqual(size_t size, const U* A, const V* B, bool* C,
                  const OffsetUtil<2>* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  ListDevicePtr<OffsetUtil<2>> d_meta(meta, 1);
  vecLessEqualKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta.get());
}

template <typename U, typename V>
void vecGreaterThan(size_t size, const U* A, const V* B, bool* C,
                    const OffsetUtil<2>* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  ListDevicePtr<OffsetUtil<2>> d_meta(meta, 1);
  vecGreaterThanKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta.get());
}

template <typename U, typename V>
void vecLessThan(size_t size, const U* A, const V* B, bool* C,
                 const OffsetUtil<2>* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  ListDevicePtr<OffsetUtil<2>> d_meta(meta, 1);
  vecLessThanKernel<U, V><<<blocks, threads>>>(size, A, B, C, d_meta.get());
}

// clang-format off
#define OPERATIONS         \
  (vecEqual)(vecNotEqual)(vecLessEqual)(vecLessThan)(vecGreaterEqual)(vecGreaterThan)

#define INSTANTIATE_COMPARISON(r, prod)                                 \
  template void BOOST_PP_SEQ_ELEM(                                      \
      0, prod)<BOOST_PP_SEQ_ELEM(1, prod), BOOST_PP_SEQ_ELEM(2, prod)>( \
      size_t, const BOOST_PP_SEQ_ELEM(1, prod)*,                        \
      const BOOST_PP_SEQ_ELEM(2, prod)*, bool*, const OffsetUtil<2>*);

#include "lamppp/tensor/supported_types.hpp"
#define TYPES_LIST LMP_TYPES()
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_COMPARISON,
                              (OPERATIONS)(TYPES_LIST)(TYPES_LIST))

#undef INSTANTIATE_COMPARISON
// clang-format on

}  // namespace lmp::tensor::detail::cuda
