#pragma once

#include "include/lamppp/autograd/functions/basic_ops.hpp"
#include "include/lamppp/autograd/functions/binary_ops.hpp"
#include "include/lamppp/autograd/variable.hpp"
#include "include/lamppp/tensor/scalar.hpp"
#include "include/lamppp/tensor/tensor_helper.hpp"

namespace autograd {

inline namespace ops {

template <Variable (*OpTag)(const Variable&, const Variable&)>
inline Variable binary_op(const Variable& a, const Variable& b) {
  return (*OpTag)(a, b);
}
template <Variable (*OpTag)(const Variable&, const Variable&)>
inline Variable binary_op(const Variable& v, Scalar s) {
  Tensor tmp = full_like(v.data(), s);
  return binary_op<OpTag>(v, Variable(tmp));
}
template <Variable (*OpTag)(const Variable&, const Variable&)>
inline Variable binary_op(Scalar s, const Variable& v) {
  Tensor tmp = full_like(v.data(), s);
  return binary_op<OpTag>(Variable(tmp), v);
}

#define DECL_BINARY_OP(op, tag)                                       \
  inline Variable operator op(const Variable& a, const Variable& b) { \
    return binary_op<&tag>(a, b);                                     \
  }                                                                   \
  inline Variable operator op(const Variable& v, Scalar s) {          \
    return binary_op<&tag>(v, s);                                     \
  }                                                                   \
  inline Variable operator op(Scalar s, const Variable& v) {          \
    return binary_op<&tag>(s, v);                                     \
  }

#define FORALL_BINARY_OPS(_) \
  _(+, add)                  \
  _(-, sub)                  \
  _(*, mul)                  \
  _(/, div)                  \
  _(==, equal)               \
  _(!=, not_equal)           \
  _(>=, greater_equal)       \
  _(<=, less_equal)          \
  _(>, greater)              \
  _(<, less)

FORALL_BINARY_OPS(DECL_BINARY_OP)

#undef FORALL_BINARY_OPS
#undef DECL_BINARY_OP

}  // namespace ops
}  // namespace autograd
