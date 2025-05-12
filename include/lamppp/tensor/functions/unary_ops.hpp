#pragma once

#include "include/lamppp/tensor/device_type.hpp"
#include "include/lamppp/tensor/dispatch_stub.hpp"
#include "include/lamppp/tensor/tensor.hpp"
#include "include/lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using relu_fn = TensorImpl (*)(const TensorImpl&);
using exp_fn = TensorImpl (*)(const TensorImpl&);
using log_fn = TensorImpl (*)(const TensorImpl&);

LMP_DECLARE_DISPATCH(relu_fn, relu_stub);
LMP_DECLARE_DISPATCH(exp_fn, exp_stub);
LMP_DECLARE_DISPATCH(log_fn, log_stub);

TensorImpl relu_cpu(const TensorImpl& a);
TensorImpl relu_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(relu_stub, DeviceType::CPU, relu_cpu);
LMP_REGISTER_DISPATCH(relu_stub, DeviceType::CUDA, relu_cuda);

TensorImpl exp_cpu(const TensorImpl& a);
TensorImpl exp_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(exp_stub, DeviceType::CPU, exp_cpu);
LMP_REGISTER_DISPATCH(exp_stub, DeviceType::CUDA, exp_cuda);

TensorImpl log_cpu(const TensorImpl& a);
TensorImpl log_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(log_stub, DeviceType::CPU, log_cpu);
LMP_REGISTER_DISPATCH(log_stub, DeviceType::CUDA, log_cuda);

Tensor relu(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);

}  // namespace lmp::tensor::ops
