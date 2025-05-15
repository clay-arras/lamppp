#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"

namespace lmp::autograd::ops {

struct EqualBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct LessBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct LessEqualBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct NotEqualBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct GreaterBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct GreaterEqualBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Equal : public ForwardFunction<Equal> {
  using DefaultBackward = EqualBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct Less : public ForwardFunction<Less> {
  using DefaultBackward = LessBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct LessEqual : public ForwardFunction<LessEqual> {
  using DefaultBackward = LessEqualBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct NotEqual : public ForwardFunction<NotEqual> {
  using DefaultBackward = NotEqualBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct Greater : public ForwardFunction<Greater> {
  using DefaultBackward = GreaterBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct GreaterEqual : public ForwardFunction<GreaterEqual> {
  using DefaultBackward = GreaterEqualBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

inline Variable equal(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Equal>({a, b})[0];
}

inline Variable not_equal(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<NotEqual>({a, b})[0];
}

inline Variable greater_equal(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<GreaterEqual>({a, b})[0];
}

inline Variable less_equal(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<LessEqual>({a, b})[0];
}

inline Variable greater(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Greater>({a, b})[0];
}

inline Variable less(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Less>({a, b})[0];
}

}  // namespace lmp::autograd::ops
