#pragma once

#ifndef _REDUCT_KERN_CUH_
#define _REDUCT_KERN_CUH_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void vecSumKernel(const float* in,
                             float* out,
                             const int* shape,
                             int* stride,
                             int axis, 
                             int outSize);
__global__ void vecMaxKernel(const float* in,
                             float* out,
                             const int* shape,
                             int* stride,
                             int axis, 
                             int outSize);

void vecSum(const float* in,
            float* out,
            const int* shape,
            int axis,
            int ndims);
void vecMax(const float* in,
            float* out,
            const int* shape,
            int axis, 
            int ndims);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _REDUCT_KERN_CUH_