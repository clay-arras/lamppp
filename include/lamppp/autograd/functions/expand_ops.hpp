#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"

namespace lmp::autograd::ops {

struct AddBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct SubtractBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct MultiplyBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct DivideBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Add : public ForwardFunction<Add> {
  using DefaultBackward = AddBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct Subtract : public ForwardFunction<Subtract> {
  using DefaultBackward = SubtractBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct Multiply : public ForwardFunction<Multiply> {
  using DefaultBackward = MultiplyBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct Divide : public ForwardFunction<Divide> {
  using DefaultBackward = DivideBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

inline Variable add(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Add>({a, b})[0];
}

inline Variable sub(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Subtract>({a, b})[0];
}

inline Variable mul(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Multiply>({a, b})[0];
}

inline Variable div(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Divide>({a, b})[0];
}

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
