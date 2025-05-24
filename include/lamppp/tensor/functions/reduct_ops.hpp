#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using sum_fn = TensorImpl (*)(const TensorImpl&, size_t axis);
using max_fn = TensorImpl (*)(const TensorImpl&, size_t axis);
using min_fn = TensorImpl (*)(const TensorImpl&, size_t axis);

LMP_DECLARE_DISPATCH(sum_fn, sum_stub);
LMP_DECLARE_DISPATCH(max_fn, max_stub);
LMP_DECLARE_DISPATCH(min_fn, min_stub);

Tensor sum(const Tensor& a, size_t axis);
Tensor max(const Tensor& a, size_t axis);
Tensor min(const Tensor& a, size_t axis);

}  // namespace lmp::tensor::ops
