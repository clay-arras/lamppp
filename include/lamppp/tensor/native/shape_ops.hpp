#pragma once

#include "lamppp/tensor/tensor.hpp"

namespace lmp::tensor::ops {

Tensor reshape(const Tensor& a, std::vector<size_t> new_shape);
Tensor squeeze(const Tensor& a, size_t dim);
Tensor expand_dims(const Tensor& a, size_t dim);

}  // namespace lmp::tensor::ops
