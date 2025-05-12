#pragma once

#include "include/lamppp/tensor/data_ptr.hpp"
#include "include/lamppp/tensor/device_type.hpp"
#include "include/lamppp/tensor/dispatch_stub.hpp"

namespace lmp::tensor::detail::native {

using empty_fn = DataPtr (*)(size_t);
DECLARE_DISPATCH(empty_fn, empty_stub);

DataPtr empty_cpu(size_t byte_size);
DataPtr empty_cuda(size_t byte_size);

REGISTER_DISPATCH(empty_stub, DeviceType::CPU, empty_cpu);
REGISTER_DISPATCH(empty_stub, DeviceType::CUDA, empty_cuda);

}  // namespace lmp::tensor::detail::native