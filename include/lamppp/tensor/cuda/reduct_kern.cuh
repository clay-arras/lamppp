#pragma once

#include <cuda_runtime.h>
#include "lamppp/tensor/align_utils.hpp"

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

template <typename T>
__global__ void vecSumKernel(const T* in, T* out, const size_t* shape,
                             stride_t* stride, size_t axis, size_t outSize);
template <typename T>
__global__ void vecMaxKernel(const T* in, T* out, const size_t* shape,
                             stride_t* stride, size_t axis, size_t outSize);
template <typename T>
__global__ void vecMinKernel(const T* in, T* out, const size_t* shape,
                             stride_t* stride, size_t axis, size_t outSize);

template <typename T>
void vecSum(const T* in, T* out, const size_t* shape, const stride_t* stride,
            size_t axis, size_t ndims, size_t size);
template <typename T>
void vecMax(const T* in, T* out, const size_t* shape, const stride_t* stride,
            size_t axis, size_t ndims, size_t size);
template <typename T>
void vecMin(const T* in, T* out, const size_t* shape, const stride_t* stride,
            size_t axis, size_t ndims, size_t size);

}  // namespace lmp::tensor::detail::cuda

#endif