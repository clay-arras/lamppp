#pragma once

#ifndef _MATRIX_KERN_CUH_
#define _MATRIX_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void cudaMatmulKernel(const float* A,
                       const float* B,
                       float* C,
                       int m,
                       int n,
                       int k);
void cudaMatMul(const float* A,
                const float* B,
                float* C,
                int m,
                int n,
                int k);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _MATRIX_KERN_CUH_
