#pragma once

#include "lamppp/tensor/data_ptr.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"

namespace lmp::tensor::detail::native {

using empty_fn = DataPtr (*)(size_t);
LMP_DECLARE_DISPATCH(empty_fn, empty_stub);

DataPtr empty_cpu(size_t byte_size);
DataPtr empty_cuda(size_t byte_size);

}  // namespace lmp::tensor::detail::native