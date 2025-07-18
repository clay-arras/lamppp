#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using conv1d_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&, size_t, size_t, size_t);
using conv2d_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&, size_t, size_t, size_t);
using conv3d_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&, size_t, size_t, size_t);

LMP_DECLARE_DISPATCH(conv1d_fn, conv1d_stub);
LMP_DECLARE_DISPATCH(conv2d_fn, conv2d_stub);
LMP_DECLARE_DISPATCH(conv3d_fn, conv3d_stub);

Tensor conv1d(const Tensor& input, const Tensor& kernel, size_t stride, size_t padding, size_t dilation);
Tensor conv2d(const Tensor& input, const Tensor& kernel, size_t stride, size_t padding, size_t dilation);
Tensor conv3d(const Tensor& input, const Tensor& kernel, size_t stride, size_t padding, size_t dilation);

}  // namespace lmp::tensor::ops