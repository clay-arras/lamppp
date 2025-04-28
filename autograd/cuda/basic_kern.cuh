#pragma once

#ifndef _BASIC_KERN_CUH_
#define _BASIC_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {

inline namespace cuda {

template <typename T>
__global__ void vecAddKernel(int size,
                             const T* A,
                             const T* B,
                             T* C);
template <typename T>
void vecAdd(int size,
            const T* A,
            const T* B,
            T* C);

template <typename T>
__global__ void vecSubKernel(int size,
                             const T* A,
                             const T* B,
                             T* C);
template <typename T>
void vecSub(int size,
            const T* A,
            const T* B,
            T* C);

template <typename T>
__global__ void vecMulKernel(int size,
                             const T* A,
                             const T* B,
                             T* C);
template <typename T>
void vecMul(int size,
            const T* A,
            const T* B,
            T* C);

template <typename T>
__global__ void vecDivKernel(int size,
                             const T* A,
                             const T* B,
                             T* C);
template <typename T>
void vecDiv(int size,
            const T* A,
            const T* B,
            T* C);


} // namespace cuda

} // namespace autograd

#endif

#endif // _BASIC_KERN_CUH_
