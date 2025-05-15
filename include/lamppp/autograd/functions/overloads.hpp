#pragma once

#include "lamppp/autograd/functions/basic_ops.hpp"
#include "lamppp/autograd/functions/binary_ops.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/tensor/scalar.hpp"
#include "lamppp/tensor/tensor_helper.hpp"

namespace lmp::autograd {

template <Variable (*OpTag)(const Variable&, const Variable&)>
inline Variable binary_op(const Variable& a, const Variable& b) {
  return (*OpTag)(a, b);
}
template <Variable (*OpTag)(const Variable&, const Variable&)>
inline Variable binary_op(const Variable& v, tensor::Scalar s) {
  tensor::Tensor tmp = full_like(v.data(), s);
  return binary_op<OpTag>(v, Variable(tmp));
}
template <Variable (*OpTag)(const Variable&, const Variable&)>
inline Variable binary_op(tensor::Scalar s, const Variable& v) {
  tensor::Tensor tmp = full_like(v.data(), s);
  return binary_op<OpTag>(Variable(tmp), v);
}

#define DECL_BINARY_OP(op, tag)                                       \
  inline Variable operator op(const Variable& a, const Variable& b) { \
    return binary_op<&tag>(a, b);                                     \
  }                                                                   \
  inline Variable operator op(const Variable& v, tensor::Scalar s) {  \
    return binary_op<&tag>(v, s);                                     \
  }                                                                   \
  inline Variable operator op(tensor::Scalar s, const Variable& v) {  \
    return binary_op<&tag>(s, v);                                     \
  }

#define FORALL_BINARY_OPS(_) \
  _(+, ops::add)             \
  _(-, ops::sub)             \
  _(*, ops::mul)             \
  _(/, ops::div)             \
  _(==, ops::equal)          \
  _(!=, ops::not_equal)      \
  _(>=, ops::greater_equal)  \
  _(<=, ops::less_equal)     \
  _(>, ops::greater)         \
  _(<, ops::less)

FORALL_BINARY_OPS(DECL_BINARY_OP)

#undef FORALL_BINARY_OPS
#undef DECL_BINARY_OP

}  // namespace lmp::autograd
