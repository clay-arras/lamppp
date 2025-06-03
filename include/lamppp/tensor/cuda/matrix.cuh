#pragma once

#include <cuda_runtime.h>

namespace lmp::tensor::detail::cuda {

/// @internal

template <typename U, typename V, typename OutType>
void cudaMatMul(const U* A, const V* B, OutType* C, size_t m, size_t n,
                size_t k);
template <typename T>
void cudaTranspose(const T* in, T* out, size_t m, size_t n);
/// @endinternal

}  // namespace lmp::tensor::detail::cuda

