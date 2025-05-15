#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using exp_fn = TensorImpl (*)(const TensorImpl&);
using log_fn = TensorImpl (*)(const TensorImpl&);
using sqrt_fn = TensorImpl (*)(const TensorImpl&);
using abs_fn = TensorImpl (*)(const TensorImpl&);
using sin_fn = TensorImpl (*)(const TensorImpl&);
using cos_fn = TensorImpl (*)(const TensorImpl&);
using tan_fn = TensorImpl (*)(const TensorImpl&);
using clamp_fn = TensorImpl (*)(const TensorImpl&, Scalar, Scalar);

LMP_DECLARE_DISPATCH(exp_fn, exp_stub);
LMP_DECLARE_DISPATCH(log_fn, log_stub);
LMP_DECLARE_DISPATCH(sqrt_fn, sqrt_stub);
LMP_DECLARE_DISPATCH(abs_fn, abs_stub);
LMP_DECLARE_DISPATCH(sin_fn, sin_stub);
LMP_DECLARE_DISPATCH(cos_fn, cos_stub);
LMP_DECLARE_DISPATCH(tan_fn, tan_stub);
LMP_DECLARE_DISPATCH(clamp_fn, clamp_stub);

TensorImpl exp_cpu(const TensorImpl& a);
TensorImpl exp_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(exp_stub, DeviceType::CPU, exp_cpu);
LMP_REGISTER_DISPATCH(exp_stub, DeviceType::CUDA, exp_cuda);

TensorImpl log_cpu(const TensorImpl& a);
TensorImpl log_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(log_stub, DeviceType::CPU, log_cpu);
LMP_REGISTER_DISPATCH(log_stub, DeviceType::CUDA, log_cuda);

TensorImpl sqrt_cpu(const TensorImpl& a);
TensorImpl sqrt_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(sqrt_stub, DeviceType::CPU, sqrt_cpu);
LMP_REGISTER_DISPATCH(sqrt_stub, DeviceType::CUDA, sqrt_cuda);

TensorImpl abs_cpu(const TensorImpl& a);
TensorImpl abs_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(abs_stub, DeviceType::CPU, abs_cpu);
LMP_REGISTER_DISPATCH(abs_stub, DeviceType::CUDA, abs_cuda);

TensorImpl sin_cpu(const TensorImpl& a);
TensorImpl sin_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(sin_stub, DeviceType::CPU, sin_cpu);
LMP_REGISTER_DISPATCH(sin_stub, DeviceType::CUDA, sin_cuda);

TensorImpl cos_cpu(const TensorImpl& a);
TensorImpl cos_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(cos_stub, DeviceType::CPU, cos_cpu);
LMP_REGISTER_DISPATCH(cos_stub, DeviceType::CUDA, cos_cuda);

TensorImpl tan_cpu(const TensorImpl& a);
TensorImpl tan_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(tan_stub, DeviceType::CPU, tan_cpu);
LMP_REGISTER_DISPATCH(tan_stub, DeviceType::CUDA, tan_cuda);

TensorImpl clamp_cpu(const TensorImpl& a, Scalar min_val, Scalar max_val);
TensorImpl clamp_cuda(const TensorImpl& a, Scalar min_val, Scalar max_val);

LMP_REGISTER_DISPATCH(clamp_stub, DeviceType::CPU, clamp_cpu);
LMP_REGISTER_DISPATCH(clamp_stub, DeviceType::CUDA, clamp_cuda);

Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& self);
Tensor abs(const Tensor& self);
Tensor sin(const Tensor& self);
Tensor cos(const Tensor& self);
Tensor tan(const Tensor& self);
Tensor clamp(const Tensor& self, Scalar min_val, Scalar max_val);

}  // namespace lmp::tensor::ops
