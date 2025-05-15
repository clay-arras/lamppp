#pragma once

#include <cuda_runtime.h>
#include "offset_util.cuh"

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

template <typename U, typename V, typename OutType>
__global__ void vecAddKernel(size_t size, const U* A, const V* B, OutType* C,
                             const OffsetUtil* meta);
template <typename U, typename V, typename OutType>
void vecAdd(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta);

template <typename U, typename V, typename OutType>
__global__ void vecSubKernel(size_t size, const U* A, const V* B, OutType* C);
template <typename U, typename V, typename OutType>
void vecSub(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta);

template <typename U, typename V, typename OutType>
__global__ void vecMulKernel(size_t size, const U* A, const V* B, OutType* C);
template <typename U, typename V, typename OutType>
void vecMul(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta);

template <typename U, typename V, typename OutType>
__global__ void vecDivKernel(size_t size, const U* A, const V* B, OutType* C);
template <typename U, typename V, typename OutType>
void vecDiv(size_t size, const U* A, const V* B, OutType* C,
            const OffsetUtil* meta);

}  // namespace lmp::tensor::detail::cuda

#endif
