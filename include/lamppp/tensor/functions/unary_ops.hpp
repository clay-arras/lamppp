#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using neg_fn = TensorImpl (*)(const TensorImpl&);
using exp_fn = TensorImpl (*)(const TensorImpl&);
using log_fn = TensorImpl (*)(const TensorImpl&);
using sqrt_fn = TensorImpl (*)(const TensorImpl&);
using abs_fn = TensorImpl (*)(const TensorImpl&);
using sin_fn = TensorImpl (*)(const TensorImpl&);
using cos_fn = TensorImpl (*)(const TensorImpl&);
using tan_fn = TensorImpl (*)(const TensorImpl&);
using clamp_fn = TensorImpl (*)(const TensorImpl&, Scalar, Scalar);

LMP_DECLARE_DISPATCH(neg_fn, neg_stub);
LMP_DECLARE_DISPATCH(exp_fn, exp_stub);
LMP_DECLARE_DISPATCH(log_fn, log_stub);
LMP_DECLARE_DISPATCH(sqrt_fn, sqrt_stub);
LMP_DECLARE_DISPATCH(abs_fn, abs_stub);
LMP_DECLARE_DISPATCH(sin_fn, sin_stub);
LMP_DECLARE_DISPATCH(cos_fn, cos_stub);
LMP_DECLARE_DISPATCH(tan_fn, tan_stub);
LMP_DECLARE_DISPATCH(clamp_fn, clamp_stub);

Tensor neg(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& self);
Tensor abs(const Tensor& self);
Tensor sin(const Tensor& self);
Tensor cos(const Tensor& self);
Tensor tan(const Tensor& self);
Tensor clamp(const Tensor& self, Scalar min_val, Scalar max_val);

}  // namespace lmp::tensor::ops
