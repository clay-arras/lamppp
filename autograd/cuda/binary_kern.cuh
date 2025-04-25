#pragma once

#ifndef _BINARY_KERN_CUH_
#define _BINARY_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void vecEqualKernel(int size,
                      const float* A,
                      const float* B,
                      float* C);
void vecEqual(int size,
              const float* A,
              const float* B,
              float* C);

__global__ void vecNotEqualKernel(int size,
                          const float* A,
                          const float* B,
                          float* C);
void vecNotEqual(int size,
                 const float* A,
                 const float* B,
                 float* C);

__global__ void vecGreaterEqualKernel(int size,
                              const float* A,
                              const float* B,
                              float* C);
void vecGreaterEqual(int size,
                     const float* A,
                     const float* B,
                     float* C);

__global__ void vecLessEqualKernel(int size,
                           const float* A,
                           const float* B,
                           float* C);
void vecLessEqual(int size,
                  const float* A,
                  const float* B,
                  float* C);

__global__ void vecGreaterThanKernel(int size,
                             const float* A,
                             const float* B,
                             float* C);
void vecGreaterThan(int size,
                    const float* A,
                    const float* B,
                    float* C);

__global__ void vecLessThanKernel(int size,
                          const float* A,
                          const float* B,
                          float* C);
void vecLessThan(int size,
                 const float* A,
                 const float* B,
                 float* C);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _BINARY_KERN_CUH_
