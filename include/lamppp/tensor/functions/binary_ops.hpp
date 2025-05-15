#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using equal_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using not_equal_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using greater_equal_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using less_equal_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using greater_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using less_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);

LMP_DECLARE_DISPATCH(equal_fn, equal_stub);
LMP_DECLARE_DISPATCH(not_equal_fn, not_equal_stub);
LMP_DECLARE_DISPATCH(greater_equal_fn, greater_equal_stub);
LMP_DECLARE_DISPATCH(less_equal_fn, less_equal_stub);
LMP_DECLARE_DISPATCH(greater_fn, greater_stub);
LMP_DECLARE_DISPATCH(less_fn, less_stub);

TensorImpl not_equal_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl not_equal_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(not_equal_stub, DeviceType::CPU, not_equal_cpu);
LMP_REGISTER_DISPATCH(not_equal_stub, DeviceType::CUDA, not_equal_cuda);

TensorImpl equal_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl equal_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(equal_stub, DeviceType::CPU, equal_cpu);
LMP_REGISTER_DISPATCH(equal_stub, DeviceType::CUDA, equal_cuda);

TensorImpl greater_equal_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl greater_equal_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(greater_equal_stub, DeviceType::CPU, greater_equal_cpu);
LMP_REGISTER_DISPATCH(greater_equal_stub, DeviceType::CUDA, greater_equal_cuda);

TensorImpl less_equal_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl less_equal_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(less_equal_stub, DeviceType::CPU, less_equal_cpu);
LMP_REGISTER_DISPATCH(less_equal_stub, DeviceType::CUDA, less_equal_cuda);

TensorImpl greater_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl greater_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(greater_stub, DeviceType::CPU, greater_cpu);
LMP_REGISTER_DISPATCH(greater_stub, DeviceType::CUDA, greater_cuda);

TensorImpl less_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl less_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(less_stub, DeviceType::CPU, less_cpu);
LMP_REGISTER_DISPATCH(less_stub, DeviceType::CUDA, less_cuda);

Tensor equal(const Tensor& a, const Tensor& b);
Tensor not_equal(const Tensor& a, const Tensor& b);
Tensor greater_equal(const Tensor& a, const Tensor& b);
Tensor less_equal(const Tensor& a, const Tensor& b);
Tensor greater(const Tensor& a, const Tensor& b);
Tensor less(const Tensor& a, const Tensor& b);

}  // namespace lmp::tensor::ops