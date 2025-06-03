#pragma once

#include "scalar.hpp"
#include "tensor.hpp"

namespace lmp::tensor {

class Tensor;

/**
 * @brief Fill a tensor with a scalar value
 * @param tensor The tensor to fill
 * @param scalar The scalar value to fill the tensor with
 * @return A new tensor with the same shape as the input tensor, filled with the scalar value
 */
Tensor full_like(const Tensor& tensor, Scalar scalar);

/**
 * @brief Fill a tensor with ones
 * @param tensor The tensor to fill
 * @return A new tensor with the same shape as the input tensor, filled with ones
 */
Tensor ones_like(const Tensor& tensor);

/**
 * @brief Fill a tensor with zeros
 * @param tensor The tensor to fill
 * @return A new tensor with the same shape as the input tensor, filled with zeros
 */
Tensor zeros_like(const Tensor& tensor);

}  // namespace lmp::tensor
