#pragma once

#ifndef _TENSOR_HELPER_H_
#define _TENSOR_HELPER_H_

#include <vector>
#include "autograd/engine/tensor.hpp"

namespace autograd {

class Tensor;

Tensor ones_like(const Tensor& tensor);
Tensor zeros_like(const Tensor& tensor);
Tensor full_like(const Tensor& tensor, Scalar scalar);

}  // namespace autograd

#endif  // _TENSOR_HELPER_H_
