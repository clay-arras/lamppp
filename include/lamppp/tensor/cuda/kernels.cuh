#pragma once

#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/functions/unary_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::cuda {

TensorImpl log_cuda(const TensorImpl& a);

LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CUDA, log_cuda);

}  // namespace lmp::tensor::cuda