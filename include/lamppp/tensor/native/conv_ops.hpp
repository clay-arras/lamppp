#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using conv_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&, size_t, size_t, size_t);

LMP_DECLARE_DISPATCH(conv_fn, conv_stub);

Tensor conv(const Tensor& input, const Tensor& kernel, size_t stride, size_t padding, size_t dilation);

}  // namespace lmp::tensor::ops