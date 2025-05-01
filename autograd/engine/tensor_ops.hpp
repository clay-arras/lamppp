#pragma once

#ifndef _TENSOR_OPS_H_
#define _TENSOR_OPS_H_

#include "tensor.hpp"

namespace autograd {
inline namespace ops {

template<auto OpTag>
inline Tensor binary_tensor_op(const Tensor& a, const Tensor& b) {
  return Tensor((a.impl_.get()->*OpTag)(*a.impl_, *b.impl_));
}

template<auto OpTag>
inline Tensor binary_tensor_op(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return binary_tensor_op<OpTag>(tensor, scalar_tensor);
}

template<auto OpTag>
inline Tensor binary_tensor_op(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return binary_tensor_op<OpTag>(scalar_tensor, tensor);
}

#define DECL_BINARY_OP(op, tag) \
  inline Tensor operator op(const Tensor& a, const Tensor& b) { \
    return binary_tensor_op<&TensorImpl::tag>(a, b); \
  } \
  inline Tensor operator op(const Tensor& tensor, float scalar) { \
    return binary_tensor_op<&TensorImpl::tag>(tensor, scalar); \
  } \
  inline Tensor operator op(float scalar, const Tensor& tensor) { \
    return binary_tensor_op<&TensorImpl::tag>(scalar, tensor); \
  }

#define FORALL_BINARY_OPS(_) \
 _(+, add) \
 _(-, sub) \
 _(*, mul) \
 _(/, div) \
 _(==, equal) \
 _(!=, not_equal) \
 _(>=, greater_equal) \
 _(<=, less_equal) \
 _(>, greater_than) \
 _(<, less_than)

FORALL_BINARY_OPS(DECL_BINARY_OP)

#undef FORALL_BINARY_OPS
#undef DECL_BINARY_OP

inline Tensor log(const Tensor& tensor) {
  return Tensor(tensor.impl_->log(*tensor.impl_));
}
inline Tensor exp(const Tensor& tensor) {
  return Tensor(tensor.impl_->exp(*tensor.impl_));
}
inline Tensor relu(const Tensor& tensor) {
  return Tensor(tensor.impl_->relu(*tensor.impl_));
}
inline Tensor matmul(const Tensor& a, const Tensor& b) {
  return Tensor(a.impl_->matmul(*a.impl_, *b.impl_));
}
inline Tensor transpose(const Tensor& tensor) {
  return Tensor(tensor.impl_->transpose(*tensor.impl_));
}
inline Tensor sum(const Tensor& tensor, int axis) {
  return Tensor(tensor.impl_->sum(*tensor.impl_, axis));
}
inline Tensor max(const Tensor& tensor, int axis) {
  return Tensor(tensor.impl_->max(*tensor.impl_, axis));
}

}  // namespace ops

}  // namespace autograd

#endif  // _TENSOR_OPS_H_