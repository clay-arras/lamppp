#pragma once

#include "autograd/engine/data_type.hpp"
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/dispatch_stub.hpp"

namespace autograd {

using copy_fn = void (*)(DeviceType, const void*, void*, size_t, DataType,
                         DataType);
DECLARE_DISPATCH(copy_fn, copy_stub);

void copy_cpu(DeviceType to_device, const void* src, void* dest, size_t size,
              DataType src_dtype, DataType dest_dtype);
void copy_cuda(DeviceType to_device, const void* src, void* dest, size_t size,
               DataType src_dtype, DataType dest_dtype);

REGISTER_DISPATCH(copy_stub, DeviceType::CPU, copy_cpu);
REGISTER_DISPATCH(copy_stub, DeviceType::CUDA, copy_cuda);

}  // namespace autograd