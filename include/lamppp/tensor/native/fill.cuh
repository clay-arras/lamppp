#pragma once

#include <cstddef>
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/scalar.hpp"

namespace lmp::tensor::detail::native {

using fill_fn = void (*)(void*, size_t, Scalar, DataType type);
LMP_DECLARE_DISPATCH(fill_fn, fill_stub);

void fill_cpu(void* ptr, size_t size, Scalar t, DataType type);
void fill_cuda(void* ptr, size_t size, Scalar t, DataType type);

}  // namespace lmp::tensor::detail::native