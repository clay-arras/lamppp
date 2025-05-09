#pragma once

#ifndef _BINARY_KERN_CUH_
#define _BINARY_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {
inline namespace cuda {

template <typename U, typename V>
__global__ void vecEqualKernel(size_t size, const U* A, const V* B, bool* C);
template <typename U, typename V>
void vecEqual(size_t size, const U* A, const V* B, bool* C);

template <typename U, typename V>
__global__ void vecNotEqualKernel(size_t size, const U* A, const V* B, bool* C);
template <typename U, typename V>
void vecNotEqual(size_t size, const U* A, const V* B, bool* C);

template <typename U, typename V>
__global__ void vecGreaterEqualKernel(size_t size, const U* A, const V* B,
                                      bool* C);
template <typename U, typename V>
void vecGreaterEqual(size_t size, const U* A, const V* B, bool* C);

template <typename U, typename V>
__global__ void vecLessEqualKernel(size_t size, const U* A, const V* B,
                                   bool* C);
template <typename U, typename V>
void vecLessEqual(size_t size, const U* A, const V* B, bool* C);

template <typename U, typename V>
__global__ void vecGreaterThanKernel(size_t size, const U* A, const V* B,
                                     bool* C);
template <typename U, typename V>
void vecGreaterThan(size_t size, const U* A, const V* B, bool* C);

template <typename U, typename V>
__global__ void vecLessThanKernel(size_t size, const U* A, const V* B, bool* C);
template <typename U, typename V>
void vecLessThan(size_t size, const U* A, const V* B, bool* C);

}  // namespace cuda
}  // namespace autograd

#endif

#endif  // _BINARY_KERN_CUH_
