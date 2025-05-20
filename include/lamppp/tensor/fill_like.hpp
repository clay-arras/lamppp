#pragma once

#include "scalar.hpp"
#include "tensor.hpp"

namespace lmp::tensor {

class Tensor;

Tensor full_like(const Tensor& tensor, Scalar scalar);
Tensor ones_like(const Tensor& tensor);
Tensor zeros_like(const Tensor& tensor);

}  // namespace lmp::tensor
