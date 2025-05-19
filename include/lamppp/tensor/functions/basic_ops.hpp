#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using add_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using sub_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using mul_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using div_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);

LMP_DECLARE_DISPATCH(add_fn, add_stub);
LMP_DECLARE_DISPATCH(sub_fn, sub_stub);
LMP_DECLARE_DISPATCH(mul_fn, mul_stub);
LMP_DECLARE_DISPATCH(div_fn, div_stub);

TensorImpl add_cpu(const TensorImpl& a, const TensorImpl& b);
// TensorImpl add_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(add_stub, DeviceType::CPU, add_cpu);
// LMP_REGISTER_DISPATCH(add_stub, DeviceType::CUDA, add_cuda);

TensorImpl sub_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl sub_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(sub_stub, DeviceType::CPU, sub_cpu);
LMP_REGISTER_DISPATCH(sub_stub, DeviceType::CUDA, sub_cuda);

TensorImpl mul_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl mul_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(mul_stub, DeviceType::CPU, mul_cpu);
LMP_REGISTER_DISPATCH(mul_stub, DeviceType::CUDA, mul_cuda);

TensorImpl div_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl div_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(div_stub, DeviceType::CPU, div_cpu);
LMP_REGISTER_DISPATCH(div_stub, DeviceType::CUDA, div_cuda);

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

}  // namespace lmp::tensor::ops