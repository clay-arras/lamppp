#pragma once

#ifndef _UNARY_KERN_CUH_
#define _UNARY_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void vecExpKernel(int size,
                    float* in,
                    float* out);
void vecExp(int size,
            const float* in,
            float* out);

__global__ void vecLogKernel(int size,
                    float* in,
                    float* out);
void vecLog(int size,
            const float* in,
            float* out);

__global__ void vecReluKernel(int size,
                     float* in,
                     float* out);
void vecRelu(int size,
             const float* in,
             float* out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _UNARY_KERN_CUH_
