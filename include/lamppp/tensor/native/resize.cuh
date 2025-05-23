#pragma once

#include <cstddef>
#include "lamppp/tensor/data_ptr.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"

namespace lmp::tensor::detail::native {

using resize_fn = void (*)(DataPtr, size_t, size_t);
LMP_DECLARE_DISPATCH(resize_fn, resize_stub);

void resize_cpu(DataPtr dptr, size_t old_byte_size, size_t new_byte_size);
void resize_cuda(DataPtr dptr, size_t old_byte_size, size_t new_byte_size);

}  // namespace lmp::tensor::detail::native
