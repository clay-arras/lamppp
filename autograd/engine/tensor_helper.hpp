#pragma once

#ifndef _TENSOR_HELPER_H_
#define _TENSOR_HELPER_H_

#include "autograd/engine/tensor.hpp"

namespace autograd {

class Tensor;

Tensor full_like(const Tensor& tensor, Scalar scalar);
Tensor ones_like(const Tensor& tensor);
Tensor zeros_like(const Tensor& tensor);

}  // namespace autograd

#endif  // _TENSOR_HELPER_H_
