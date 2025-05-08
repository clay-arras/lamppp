#pragma once

#ifndef _BASIC_KERN_CUH_
#define _BASIC_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {

inline namespace cuda {

template <typename U, typename V, typename OutType>
__global__ void vecAddKernel(size_t size, const U* A, const V* B, OutType* C);
template <typename U, typename V, typename OutType>
void vecAdd(size_t size, const U* A, const V* B, OutType* C);

template <typename U, typename V, typename OutType>
__global__ void vecSubKernel(size_t size, const U* A, const V* B, OutType* C);
template <typename U, typename V, typename OutType>
void vecSub(size_t size, const U* A, const V* B, OutType* C);

template <typename U, typename V, typename OutType>
__global__ void vecMulKernel(size_t size, const U* A, const V* B, OutType* C);
template <typename U, typename V, typename OutType>
void vecMul(size_t size, const U* A, const V* B, OutType* C);

template <typename U, typename V, typename OutType>
__global__ void vecDivKernel(size_t size, const U* A, const V* B, OutType* C);
template <typename U, typename V, typename OutType>
void vecDiv(size_t size, const U* A, const V* B, OutType* C);

}  // namespace cuda

}  // namespace autograd

#endif

#endif  // _BASIC_KERN_CUH_
