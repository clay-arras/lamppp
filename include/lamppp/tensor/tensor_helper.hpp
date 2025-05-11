#pragma once

#include "scalar.hpp"
#include "tensor.hpp"

namespace autograd {

class Tensor;

Tensor full_like(const Tensor& tensor, Scalar scalar);
Tensor ones_like(const Tensor& tensor);
Tensor zeros_like(const Tensor& tensor);

}  // namespace autograd
