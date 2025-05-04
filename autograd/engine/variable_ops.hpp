#pragma once

#ifndef _VARIABLE_OPS_H_
#define _VARIABLE_OPS_H_

#include "autograd/engine/functions/basic_ops.hpp"
#include "autograd/engine/functions/binary_ops.hpp"
#include "autograd/engine/functions/matrix_ops.hpp"
#include "autograd/engine/functions/reduct_ops.hpp"
#include "autograd/engine/functions/unary_ops.hpp"
#include "autograd/engine/scalar.hpp"
#include "variable.hpp"
#include "tensor_helper.hpp"

namespace autograd {
inline namespace ops {

struct VariableOpFact {
  template <typename Op, typename... Args>
  static variable_list apply(variable_list variables, Args&&... args) {
    Op op_fn(std::forward<Args>(args)...);
    variable_list result =
        op_fn.template apply<Args...>(variables, std::forward<Args>(args)...);
    return result;
  }
};

template <class OpTag>
inline Variable binary_op(Variable const& a, Variable const& b) {
  return VariableOpFact::apply<OpTag>({a, b})[0];
}
template <class OpTag>
inline Variable binary_op(Variable const& v, Scalar s) {
  Tensor tmp = full_like(v.data(), s);
  return binary_op<OpTag>(v, Variable(tmp));
}
template <class OpTag>
inline Variable binary_op(Scalar s, Variable const& v) {
  Tensor tmp = full_like(v.data(), s);
  return binary_op<OpTag>(Variable(tmp), v);
}

#define DECL_BINARY_OP(op, tag)                                       \
  inline Variable operator op(Variable const& a, Variable const& b) { \
    return binary_op<tag>(a, b);                                      \
  }                                                                   \
  inline Variable operator op(Variable const& v, Scalar s) {          \
    return binary_op<tag>(v, s);                                      \
  }                                                                   \
  inline Variable operator op(Scalar s, Variable const& v) {          \
    return binary_op<tag>(s, v);                                      \
  }

#define FORALL_BINARY_OPS(_) \
  _(+, Add)                  \
  _(-, Subtract)             \
  _(*, Multiply)             \
  _(/, Divide)               \
  _(==, Equal)               \
  _(!=, NotEqual)            \
  _(>=, GreaterEqual)        \
  _(<=, LessEqual)           \
  _(>, Greater)              \
  _(<, Less)

FORALL_BINARY_OPS(DECL_BINARY_OP)

#undef FORALL_BINARY_OPS
#undef DECL_BINARY_OP

inline Variable matmul(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<MatrixMultiplication>({a, b})[0];
}

inline Variable transpose(const Variable& a) {
  return VariableOpFact::apply<Transpose>({a})[0];
}

inline Variable sum(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Summation>({a}, axis)[0];
}

inline Variable max(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Maximum>({a}, axis)[0];
}

inline Variable exp(const Variable& a) {
  return VariableOpFact::apply<Exponential>({a})[0];
}

inline Variable log(const Variable& a) {
  return VariableOpFact::apply<Logarithm>({a})[0];
}

inline Variable relu(const Variable& a) {
  return VariableOpFact::apply<ReLU>({a})[0];
}

}  // namespace ops
}  // namespace autograd

#endif  // _VARIABLE_OPS_H_