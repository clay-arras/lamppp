#pragma once

#ifndef _UNARY_KERN_CUH_
#define _UNARY_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {
inline namespace cuda {

template <typename T>
__global__ void vecExpKernel(size_t size,
                    T* in,
                    T* out);
template <typename T>
void vecExp(size_t size,
            const T* in,
            T* out);

template <typename T>
__global__ void vecLogKernel(size_t size,
                    T* in,
                    T* out);
template <typename T>
void vecLog(size_t size,
            const T* in,
            T* out);

template <typename T>
__global__ void vecReluKernel(size_t size,
                     T* in,
                     T* out);
template <typename T>
void vecRelu(size_t size,
             const T* in,
             T* out);

} // namespace cuda

} // namespace autograd

#endif

#endif // _UNARY_KERN_CUH_
