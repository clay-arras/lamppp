#pragma once
#ifndef _TENSOR_OPS_H_
#define _TENSOR_OPS_H_

#include "tensor.h"

namespace autograd {
inline namespace ops {

Tensor operator+(const Tensor& tensor, float scalar);
Tensor operator-(const Tensor& tensor, float scalar);
Tensor operator*(const Tensor& tensor, float scalar);
Tensor operator/(const Tensor& tensor, float scalar);

Tensor operator+(float scalar, const Tensor& tensor);
Tensor operator-(float scalar, const Tensor& tensor);
Tensor operator*(float scalar, const Tensor& tensor);
Tensor operator/(float scalar, const Tensor& tensor);

bool operator==(const Tensor& lhs, const Tensor& rhs);

} // namespace ops
} // namespace autograd

#endif // _TENSOR_OPS_H_ 