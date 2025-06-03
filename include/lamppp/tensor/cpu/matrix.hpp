#pragma once

#include <cstddef>

namespace lmp::tensor::detail::cpu {

/// @internal
template <typename U, typename V, typename OutType>
void cpuMatmulKernel(const U* A, const V* B, OutType* C, size_t m, size_t n,
                     size_t k);

template <typename U, typename V, typename OutType>
void cpuMatMul(const U* A, const V* B, OutType* C, size_t m, size_t n,
               size_t k);

template <typename T>
void cpuTransposeKernel(const T* in, T* out, size_t m, size_t n);

template <typename T>
void cpuTranspose(const T* in, T* out, size_t m, size_t n);
/// @endinternal

}  // namespace lmp::tensor::detail::cpu
