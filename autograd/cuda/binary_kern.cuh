#pragma once

#ifndef _BINARY_KERN_CUH_
#define _BINARY_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {
inline namespace cuda {

template <typename T>
__global__ void vecEqualKernel(int size,
                      const T* A,
                      const T* B,
                      T* C);
template <typename T>
void vecEqual(int size,
              const T* A,
              const T* B,
              T* C);

template <typename T>
__global__ void vecNotEqualKernel(int size,
                          const T* A,
                          const T* B,
                          T* C);
template <typename T>
void vecNotEqual(int size,
                 const T* A,
                 const T* B,
                 T* C);

template <typename T>
__global__ void vecGreaterEqualKernel(int size,
                              const T* A,
                              const T* B,
                              T* C);
template <typename T>
void vecGreaterEqual(int size,
                     const T* A,
                     const T* B,
                     T* C);

template <typename T>
__global__ void vecLessEqualKernel(int size,
                           const T* A,
                           const T* B,
                           T* C);
template <typename T>
void vecLessEqual(int size,
                  const T* A,
                  const T* B,
                  T* C);

template <typename T>
__global__ void vecGreaterThanKernel(int size,
                             const T* A,
                             const T* B,
                             T* C);
template <typename T>
void vecGreaterThan(int size,
                    const T* A,
                    const T* B,
                    T* C);

template <typename T>
__global__ void vecLessThanKernel(int size,
                          const T* A,
                          const T* B,
                          T* C);
template <typename T>
void vecLessThan(int size,
                 const T* A,
                 const T* B,
                 T* C);

} // namespace cuda
} // namespace autograd

#endif

#endif // _BINARY_KERN_CUH_
