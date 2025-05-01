#pragma once

#ifndef _MATRIX_KERN_CUH_
#define _MATRIX_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {
inline namespace cuda {

template <typename T>
__global__ void cudaMatmulKernel(const T* A,
                       const T* B,
                       T* C,
                       size_t m,
                       size_t n,
                       size_t k);

template <typename T>
void cudaMatMul(const T* A,
                const T* B,
                T* C,
                size_t m,
                size_t n,
                size_t k);

template <typename T>
__global__ void cudaTransposeKernel(const T* in,
                                    T* out,
                                    size_t m,
                                    size_t n); 

template <typename T>
void cudaTranspose(const T* in,
                              T* out,
                              size_t m,
                              size_t n);

} // namespace cuda
} // namespace autograd

#endif

#endif // _MATRIX_KERN_CUH_
