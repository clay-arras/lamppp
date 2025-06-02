#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using matmul_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using transpose_fn = TensorImpl (*)(const TensorImpl&);

LMP_DECLARE_DISPATCH(matmul_fn, matmul_stub);
LMP_DECLARE_DISPATCH(transpose_fn, transpose_stub);

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& a);

}  // namespace lmp::tensor::ops
