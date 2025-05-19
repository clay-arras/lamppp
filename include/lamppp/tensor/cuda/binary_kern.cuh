#pragma once

#include <cuda_runtime.h>
#include "offset_util.cuh"

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

template <typename U, typename V>
__global__ void vecEqualKernel(size_t size, const U* A, const V* B, bool* C,
                               const OffsetUtil<2>* meta);
template <typename U, typename V>
void vecEqual(size_t size, const U* A, const V* B, bool* C,
              const OffsetUtil<2>* meta);

template <typename U, typename V>
__global__ void vecNotEqualKernel(size_t size, const U* A, const V* B, bool* C,
                                  const OffsetUtil<2>* meta);
template <typename U, typename V>
void vecNotEqual(size_t size, const U* A, const V* B, bool* C,
                 const OffsetUtil<2>* meta);

template <typename U, typename V>
__global__ void vecGreaterEqualKernel(size_t size, const U* A, const V* B,
                                      bool* C, const OffsetUtil<2>* meta);
template <typename U, typename V>
void vecGreaterEqual(size_t size, const U* A, const V* B, bool* C,
                     const OffsetUtil<2>* meta);

template <typename U, typename V>
__global__ void vecLessEqualKernel(size_t size, const U* A, const V* B, bool* C,
                                   const OffsetUtil<2>* meta);
template <typename U, typename V>
void vecLessEqual(size_t size, const U* A, const V* B, bool* C,
                  const OffsetUtil<2>* meta);

template <typename U, typename V>
__global__ void vecGreaterThanKernel(size_t size, const U* A, const V* B,
                                     bool* C, const OffsetUtil<2>* meta);
template <typename U, typename V>
void vecGreaterThan(size_t size, const U* A, const V* B, bool* C,
                    const OffsetUtil<2>* meta);

template <typename U, typename V>
__global__ void vecLessThanKernel(size_t size, const U* A, const V* B, bool* C,
                                  const OffsetUtil<2>* meta);
template <typename U, typename V>
void vecLessThan(size_t size, const U* A, const V* B, bool* C,
                 const OffsetUtil<2>* meta);

}  // namespace lmp::tensor::detail::cuda

#endif
