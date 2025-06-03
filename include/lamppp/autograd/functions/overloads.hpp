#pragma once

#include "lamppp/autograd/functions/expand_ops.hpp"
#include "lamppp/autograd/functions/unary_ops.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/tensor/fill_like.hpp"
#include "lamppp/tensor/scalar.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::autograd {

template <Variable (*OpTag)(const Variable&, const Variable&)>
inline Variable binary_op(const Variable& a, const Variable& b) {
  return (*OpTag)(a, b);
}
template <Variable (*OpTag)(const Variable&, const Variable&)>
inline Variable binary_op(const Variable& v, tensor::Scalar s) {
  tensor::Tensor scalar_tensor(std::vector<tensor::Scalar>(1, s), {1}, v.data().device(),
                        v.data().type());  // rely on broadcasting
  return binary_op<OpTag>(v, Variable(scalar_tensor));
}
template <Variable (*OpTag)(const Variable&, const Variable&)>
inline Variable binary_op(tensor::Scalar s, const Variable& v) {
  tensor::Tensor scalar_tensor(std::vector<tensor::Scalar>(1, s), {1}, v.data().device(),
                        v.data().type());  // rely on broadcasting
  return binary_op<OpTag>(Variable(scalar_tensor), v);
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
  _(==, ops::eq)             \
  _(!=, ops::ne)             \
  _(>=, ops::ge)             \
  _(<=, ops::le)             \
  _(>, ops::gt)              \
  _(<, ops::lt)

FORALL_BINARY_OPS(DECL_BINARY_OP)

#undef FORALL_BINARY_OPS
#undef DECL_BINARY_OP

inline Variable operator-(const Variable& a) {  return ops::neg(a); }

}  // namespace lmp::autograd
