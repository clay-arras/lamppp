#pragma once

#include "include/lamppp/tensor/device_type.hpp"
#include "include/lamppp/tensor/dispatch_stub.hpp"
#include "include/lamppp/tensor/tensor.hpp"
#include "include/lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using sum_fn = TensorImpl (*)(const TensorImpl&, size_t axis);
using mean_fn = TensorImpl (*)(const TensorImpl&, size_t axis);
using max_fn = TensorImpl (*)(const TensorImpl&, size_t axis);
using min_fn = TensorImpl (*)(const TensorImpl&, size_t axis);

DECLARE_DISPATCH(sum_fn, sum_stub);
DECLARE_DISPATCH(mean_fn, mean_stub);
DECLARE_DISPATCH(max_fn, max_stub);
DECLARE_DISPATCH(min_fn, min_stub);

TensorImpl sum_cpu(const TensorImpl& a, size_t axis);
TensorImpl sum_cuda(const TensorImpl& a, size_t axis);

TensorImpl mean_cpu(const TensorImpl& a, size_t axis);
TensorImpl mean_cuda(const TensorImpl& a, size_t axis);

TensorImpl max_cpu(const TensorImpl& a, size_t axis);
TensorImpl max_cuda(const TensorImpl& a, size_t axis);

TensorImpl min_cpu(const TensorImpl& a, size_t axis);
TensorImpl min_cuda(const TensorImpl& a, size_t axis);

REGISTER_DISPATCH(sum_stub, DeviceType::CPU, sum_cpu);
REGISTER_DISPATCH(sum_stub, DeviceType::CUDA, sum_cuda);

REGISTER_DISPATCH(mean_stub, DeviceType::CPU, mean_cpu);
REGISTER_DISPATCH(mean_stub, DeviceType::CUDA, mean_cuda);

REGISTER_DISPATCH(max_stub, DeviceType::CPU, max_cpu);
REGISTER_DISPATCH(max_stub, DeviceType::CUDA, max_cuda);

REGISTER_DISPATCH(min_stub, DeviceType::CPU, min_cpu);
REGISTER_DISPATCH(min_stub, DeviceType::CUDA, min_cuda);

Tensor sum(const Tensor& a, size_t axis);
Tensor mean(const Tensor& a, size_t axis);
Tensor max(const Tensor& a, size_t axis);
Tensor min(const Tensor& a, size_t axis);

}  // namespace lmp::tensor::ops
