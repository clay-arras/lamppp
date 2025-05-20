#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

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

}  // namespace lmp::tensor::detail::cuda

#endif
