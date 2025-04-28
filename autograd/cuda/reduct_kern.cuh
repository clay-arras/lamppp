#pragma once

#ifndef _REDUCT_KERN_CUH_
#define _REDUCT_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {
inline namespace cuda {

template <typename T>
__global__ void vecSumKernel(const T* in,
                             T* out,
                             const int* shape,
                             int* stride,
                             int axis, 
                             int outSize);
template <typename T>
__global__ void vecMaxKernel(const T* in,
                             T* out,
                             const int* shape,
                             int* stride,
                             int axis, 
                             int outSize);

template <typename T>
void vecSum(const T* in,
            T* out,
            const int* shape,
            int axis,
            int ndims);
template <typename T>
void vecMax(const T* in,
            T* out,
            const int* shape,
            int axis, 
            int ndims);

} // namespace cuda
} // namespace autograd

#endif

#endif // _REDUCT_KERN_CUH_