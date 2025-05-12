#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

template <typename T>
__global__ void vecExpKernel(size_t size, T* in, T* out);
template <typename T>
void vecExp(size_t size, const T* in, T* out);

template <typename T>
__global__ void vecLogKernel(size_t size, T* in, T* out);
template <typename T>
void vecLog(size_t size, const T* in, T* out);

template <typename T>
__global__ void vecReluKernel(size_t size, T* in, T* out);
template <typename T>
void vecRelu(size_t size, const T* in, T* out);

}  // namespace lmp::tensor::detail::cuda

#endif
