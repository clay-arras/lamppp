#pragma once

#include "autograd/engine/device_type.hpp"
#include "autograd/engine/dispatch_stub.hpp"

namespace autograd {

using copy_fn = void (*)(void*, void*, size_t, DeviceType);
DECLARE_DISPATCH(copy_fn, copy_stub);

void copy_cpu(void* src, void* dest, size_t size, DeviceType to_device);
void copy_cuda(void* src, void* dest, size_t size, DeviceType to_device);

REGISTER_DISPATCH(copy_stub, DeviceType::CPU, copy_cpu);
REGISTER_DISPATCH(copy_stub, DeviceType::CUDA, copy_cuda);

}  // namespace autograd