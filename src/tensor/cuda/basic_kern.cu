#include <linux/limits.h>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "lamppp/tensor/cuda/basic_kern.cuh"
#include "lamppp/tensor/cuda/list_ptr.cuh"

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

  ListDevicePtr<OffsetUtil> d_meta(meta, 1);
  vecAddKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C, d_meta.get());
}

template <typename U, typename V, typename OutType>
void vecSub(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  ListDevicePtr<OffsetUtil> d_meta(meta, 1);
  vecSubKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C, d_meta.get());
}

template <typename U, typename V, typename OutType>
void vecMul(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  ListDevicePtr<OffsetUtil> d_meta(meta, 1);
  vecMulKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C, d_meta.get());
}

template <typename U, typename V, typename OutType>
void vecDiv(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  ListDevicePtr<OffsetUtil> d_meta(meta, 1);
  vecDivKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C, d_meta.get());
}

// clang-format off
#define OPERATIONS (vecAdd)(vecSub)(vecMul)(vecDiv)

#define INSTANTIATE(r, prod)                                           \
  template void BOOST_PP_SEQ_ELEM(                                     \
      0, prod)<BOOST_PP_SEQ_ELEM(1, prod), BOOST_PP_SEQ_ELEM(2, prod), \
               BOOST_PP_SEQ_ELEM(3, prod)>(                            \
      size_t, const BOOST_PP_SEQ_ELEM(1, prod)*,                       \
      const BOOST_PP_SEQ_ELEM(2, prod)*, BOOST_PP_SEQ_ELEM(3, prod)*,  \
      const OffsetUtil*);

#include "lamppp/tensor/supported_types.hpp"
#define TYPES_LIST LMP_TYPES()
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
                              (OPERATIONS)(TYPES_LIST)(TYPES_LIST)(TYPES_LIST))

#undef INSTANTIATE
// clang-format on

}  // namespace lmp::tensor::detail::cuda