#pragma once

#ifndef _BASIC_KERN_CUH_
#define _BASIC_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {

inline namespace cuda {

template <typename T>
__global__ void vecAddKernel(size_t size,
                             const T* A,
                             const T* B,
                             T* C);
template <typename T>
void vecAdd(size_t size,
            const T* A,
            const T* B,
            T* C);

template <typename T>
__global__ void vecSubKernel(size_t size,
                             const T* A,
                             const T* B,
                             T* C);
template <typename T>
void vecSub(size_t size,
            const T* A,
            const T* B,
            T* C);

template <typename T>
__global__ void vecMulKernel(size_t size,
                             const T* A,
                             const T* B,
                             T* C);
template <typename T>
void vecMul(size_t size,
            const T* A,
            const T* B,
            T* C);

template <typename T>
__global__ void vecDivKernel(size_t size,
                             const T* A,
                             const T* B,
                             T* C);
template <typename T>
void vecDiv(size_t size,
            const T* A,
            const T* B,
            T* C);


} // namespace cuda

} // namespace autograd

#endif

#endif // _BASIC_KERN_CUH_
