#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {
inline namespace cuda {

template <typename T>
__global__ void vecSumKernel(const T* in, T* out, const size_t* shape,
                             size_t* stride, size_t axis, size_t outSize);
template <typename T>
__global__ void vecMaxKernel(const T* in, T* out, const size_t* shape,
                             size_t* stride, size_t axis, size_t outSize);

template <typename T>
void vecSum(const T* in, T* out, const size_t* shape, size_t axis,
            size_t ndims);
template <typename T>
void vecMax(const T* in, T* out, const size_t* shape, size_t axis,
            size_t ndims);

}  // namespace cuda
}  // namespace autograd

#endif