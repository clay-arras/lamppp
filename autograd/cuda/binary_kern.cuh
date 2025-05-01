#pragma once

#ifndef _BINARY_KERN_CUH_
#define _BINARY_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace autograd {
inline namespace cuda {

template <typename T>
__global__ void vecEqualKernel(size_t size,
                      const T* A,
                      const T* B,
                      T* C);
template <typename T>
void vecEqual(size_t size,
              const T* A,
              const T* B,
              T* C);

template <typename T>
__global__ void vecNotEqualKernel(size_t size,
                          const T* A,
                          const T* B,
                          T* C);
template <typename T>
void vecNotEqual(size_t size,
                 const T* A,
                 const T* B,
                 T* C);

template <typename T>
__global__ void vecGreaterEqualKernel(size_t size,
                              const T* A,
                              const T* B,
                              T* C);
template <typename T>
void vecGreaterEqual(size_t size,
                     const T* A,
                     const T* B,
                     T* C);

template <typename T>
__global__ void vecLessEqualKernel(size_t size,
                           const T* A,
                           const T* B,
                           T* C);
template <typename T>
void vecLessEqual(size_t size,
                  const T* A,
                  const T* B,
                  T* C);

template <typename T>
__global__ void vecGreaterThanKernel(size_t size,
                             const T* A,
                             const T* B,
                             T* C);
template <typename T>
void vecGreaterThan(size_t size,
                    const T* A,
                    const T* B,
                    T* C);

template <typename T>
__global__ void vecLessThanKernel(size_t size,
                          const T* A,
                          const T* B,
                          T* C);
template <typename T>
void vecLessThan(size_t size,
                 const T* A,
                 const T* B,
                 T* C);

} // namespace cuda
} // namespace autograd

#endif

#endif // _BINARY_KERN_CUH_
