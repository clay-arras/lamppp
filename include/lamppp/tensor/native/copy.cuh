#pragma once

#include "cuda_runtime.h"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"

namespace lmp::tensor::detail::native {

using copy_fn = void (*)(DeviceType, const void*, void*, size_t, DataType,
                         DataType);
LMP_DECLARE_DISPATCH(copy_fn, copy_stub);

void copy_cpu(DeviceType to_device, const void* src, void* dest, size_t size,
              DataType src_dtype, DataType dest_dtype);
void copy_cuda(DeviceType to_device, const void* src, void* dest, size_t size,
               DataType src_dtype, DataType dest_dtype);

template <typename U, typename V>
__global__ void cudaVecCopyKernel(size_t size, const U* in, V* out);

template <typename U, typename V>
void cudaVecCopy(size_t size, const U* in, V* out);

template <typename U, typename V>
void vecCopy(size_t size, const U* in, V* out);

}  // namespace lmp::tensor::detail::native