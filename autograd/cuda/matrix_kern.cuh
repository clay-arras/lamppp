#pragma once

#ifndef _MATRIX_KERN_CUH_
#define _MATRIX_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {
inline namespace cuda {

template <typename U, typename V, typename OutType>
__global__ void cudaMatmulKernel(const U* A, const V* B, OutType* C, size_t m,
                                 size_t n, size_t k);

template <typename U, typename V, typename OutType>
void cudaMatMul(const U* A, const V* B, OutType* C, size_t m, size_t n,
                size_t k);

template <typename T>
__global__ void cudaTransposeKernel(const T* in, T* out, size_t m, size_t n);

template <typename T>
void cudaTranspose(const T* in, T* out, size_t m, size_t n);

}  // namespace cuda
}  // namespace autograd

#endif

#endif  // _MATRIX_KERN_CUH_
