#pragma once

#include <cstddef>
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/dispatch_stub.hpp"
#include "autograd/engine/scalar.hpp"

namespace autograd {

using fill_fn = void (*)(void*, size_t, Scalar, DataType type);
DECLARE_DISPATCH(fill_fn, fill_stub);

void fill_cpu(void* ptr, size_t size, Scalar t, DataType type);
void fill_cuda(void* ptr, size_t size, Scalar t, DataType type);

REGISTER_DISPATCH(fill_stub, DeviceType::CPU, fill_cpu);
REGISTER_DISPATCH(fill_stub, DeviceType::CUDA, fill_cuda);

}  // namespace autograd