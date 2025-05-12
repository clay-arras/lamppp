#pragma once

#include "include/lamppp/tensor/device_type.hpp"
#include "include/lamppp/tensor/dispatch_stub.hpp"
#include "include/lamppp/tensor/tensor.hpp"
#include "include/lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using matmul_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using transpose_fn = TensorImpl (*)(const TensorImpl&);

LMP_DECLARE_DISPATCH(matmul_fn, matmul_stub);
LMP_DECLARE_DISPATCH(transpose_fn, transpose_stub);

TensorImpl matmul_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl matmul_cuda(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(matmul_stub, DeviceType::CPU, matmul_cpu);
LMP_REGISTER_DISPATCH(matmul_stub, DeviceType::CUDA, matmul_cuda);

TensorImpl transpose_cpu(const TensorImpl& a);
TensorImpl transpose_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(transpose_stub, DeviceType::CPU, transpose_cpu);
LMP_REGISTER_DISPATCH(transpose_stub, DeviceType::CUDA, transpose_cuda);

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& a);

}  // namespace lmp::tensor::ops
