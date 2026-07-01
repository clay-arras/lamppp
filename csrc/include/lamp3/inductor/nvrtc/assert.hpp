#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "lamp3/common/assert.hpp"

#define NVRTC_CHECK(x)                                      \
  do {                                                      \
    const nvrtcResult nvrtc_r = (x);                        \
    LMP_CHECK(nvrtc_r == NVRTC_SUCCESS)                     \
        << "NVRTC error: " << nvrtcGetErrorString(nvrtc_r); \
  } while (0)

#define CU_CHECK(x)                                                     \
  do {                                                                  \
    const CUresult cu_r = (x);                                          \
    if (cu_r != CUDA_SUCCESS) {                                         \
      const char* cu_s = nullptr;                                       \
      cuGetErrorString(cu_r, &cu_s);                                    \
      LMP_CHECK(false) << "CUDA driver error: " << (cu_s ? cu_s : "?"); \
    }                                                                   \
  } while (0)

#define CUDART_CHECK(x)                                            \
  do {                                                             \
    const cudaError_t cudart_r = (x);                              \
    LMP_CHECK(cudart_r == cudaSuccess)                             \
        << "CUDA runtime error: " << cudaGetErrorString(cudart_r); \
  } while (0)
