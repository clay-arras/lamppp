#pragma once

#include "autograd/engine/data_ptr.hpp"
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/dispatch_stub.hpp"

namespace autograd {

using copy_fn = DataPtr (*)(DataPtr, size_t, DeviceType);
DECLARE_DISPATCH(copy_fn, copy_stub);

DataPtr copy_cpu(DataPtr src, size_t size, DeviceType to_device);
DataPtr copy_cuda(DataPtr src, size_t size, DeviceType to_device);

REGISTER_DISPATCH(copy_stub, DeviceType::CPU, copy_cpu);
REGISTER_DISPATCH(copy_stub, DeviceType::CUDA, copy_cuda);

}  // namespace autograd