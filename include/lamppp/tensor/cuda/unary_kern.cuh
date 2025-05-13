#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

template <typename T>
__global__ void vecExpKernel(T* in, T* out, size_t size);
template <typename T>
void vecExp(const T* in, T* out, size_t size);

template <typename T>
__global__ void vecLogKernel(T* in, T* out, size_t size);
template <typename T>
void vecLog(const T* in, T* out, size_t size);

template <typename T>
__global__ void vecSqrtKernel(const T* in, T* out, size_t size);
template <typename T>
void vecSqrt(const T* in, T* out, size_t size);

template <typename T>
__global__ void vecAbsKernel(const T* in, T* out, size_t size);
template <typename T>
void vecAbs(const T* in, T* out, size_t size);

template <typename T>
__global__ void vecSinKernel(const T* in, T* out, size_t size);
template <typename T>
void vecSin(const T* in, T* out, size_t size);

template <typename T>
__global__ void vecCosKernel(const T* in, T* out, size_t size);
template <typename T>
void vecCos(const T* in, T* out, size_t size);

template <typename T>
__global__ void vecTanKernel(const T* in, T* out, size_t size);
template <typename T>
void vecTan(const T* in, T* out, size_t size);

template <typename T>
__global__ void vecClampKernel(const T* in, T min_val, T max_val, T* out,
                               size_t size);
template <typename T>
void vecClamp(const T* in, T min_val, T max_val, T* out, size_t size);

}  // namespace lmp::tensor::detail::cuda

#endif
