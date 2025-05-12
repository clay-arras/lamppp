#pragma once

#include "cuda_runtime.h"
#include "include/lamppp/tensor/data_type.hpp"
#include "include/lamppp/tensor/device_type.hpp"
#include "include/lamppp/tensor/dispatch_stub.hpp"

namespace lmp::tensor::detail::native {

using copy_fn = void (*)(DeviceType, const void*, void*, size_t, DataType,
                         DataType);
DECLARE_DISPATCH(copy_fn, copy_stub);

void copy_cpu(DeviceType to_device, const void* src, void* dest, size_t size,
              DataType src_dtype, DataType dest_dtype);
void copy_cuda(DeviceType to_device, const void* src, void* dest, size_t size,
               DataType src_dtype, DataType dest_dtype);

REGISTER_DISPATCH(copy_stub, DeviceType::CPU, copy_cpu);
REGISTER_DISPATCH(copy_stub, DeviceType::CUDA, copy_cuda);

template <typename U, typename V>
__global__ void vecCopyKernel(size_t size, const U* in, V* out);

template <typename U, typename V>
void vecCopy(size_t size, const U* in, V* out);

}  // namespace lmp::tensor::detail::native