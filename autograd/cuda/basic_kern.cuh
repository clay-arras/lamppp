#pragma once

#ifndef _BASIC_KERN_CUH_
#define _BASIC_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void vecAddKernel(int size,
                             const float* A,
                             const float* B,
                             float* C);
void vecAdd(int size,
            const float* A,
            const float* B,
            float* C);

__global__ void vecSubKernel(int size,
                             const float* A,
                             const float* B,
                             float* C);
void vecSub(int size,
            const float* A,
            const float* B,
            float* C);

__global__ void vecMulKernel(int size,
                             const float* A,
                             const float* B,
                             float* C);
void vecMul(int size,
            const float* A,
            const float* B,
            float* C);

__global__ void vecDivKernel(int size,
                             const float* A,
                             const float* B,
                             float* C);
void vecDiv(int size,
            const float* A,
            const float* B,
            float* C);

#ifdef __cplusplus

} // extern "C"

#endif

#endif // _BASIC_KERN_CUH_
