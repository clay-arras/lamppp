#pragma once

#ifndef _UNARY_KERN_CUH_
#define _UNARY_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {
inline namespace cuda {

template <typename T>
__global__ void vecExpKernel(int size,
                    T* in,
                    T* out);
template <typename T>
void vecExp(int size,
            const T* in,
            T* out);

template <typename T>
__global__ void vecLogKernel(int size,
                    T* in,
                    T* out);
template <typename T>
void vecLog(int size,
            const T* in,
            T* out);

template <typename T>
__global__ void vecReluKernel(int size,
                     T* in,
                     T* out);
template <typename T>
void vecRelu(int size,
             const T* in,
             T* out);

} // namespace cuda

} // namespace autograd

#endif

#endif // _UNARY_KERN_CUH_
