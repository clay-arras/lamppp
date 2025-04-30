#pragma once

#ifndef _TENSOR_OPS_H_
#define _TENSOR_OPS_H_

#include "tensor.hpp"

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

Tensor operator==(const Tensor& tensor, float scalar);
Tensor operator!=(const Tensor& tensor, float scalar);
Tensor operator>=(const Tensor& tensor, float scalar);
Tensor operator<=(const Tensor& tensor, float scalar);
Tensor operator>(const Tensor& tensor, float scalar);
Tensor operator<(const Tensor& tensor, float scalar);

Tensor operator==(float scalar, const Tensor& tensor);
Tensor operator!=(float scalar, const Tensor& tensor);
Tensor operator>=(float scalar, const Tensor& tensor);
Tensor operator<=(float scalar, const Tensor& tensor);
Tensor operator>(float scalar, const Tensor& tensor);
Tensor operator<(float scalar, const Tensor& tensor);

}  // namespace ops

}  // namespace autograd

#endif  // _TENSOR_OPS_H_