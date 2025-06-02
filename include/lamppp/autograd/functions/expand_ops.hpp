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

struct PowerBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Add : public ForwardFunction<Add> {
  using DefaultBackward = AddBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Subtract : public ForwardFunction<Subtract> {
  using DefaultBackward = SubtractBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Multiply : public ForwardFunction<Multiply> {
  using DefaultBackward = MultiplyBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Divide : public ForwardFunction<Divide> {
  using DefaultBackward = DivideBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Power : public ForwardFunction<Power> {
  using DefaultBackward = PowerBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

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
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Less : public ForwardFunction<Less> {
  using DefaultBackward = LessBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct LessEqual : public ForwardFunction<LessEqual> {
  using DefaultBackward = LessEqualBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct NotEqual : public ForwardFunction<NotEqual> {
  using DefaultBackward = NotEqualBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Greater : public ForwardFunction<Greater> {
  using DefaultBackward = GreaterBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct GreaterEqual : public ForwardFunction<GreaterEqual> {
  using DefaultBackward = GreaterEqualBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
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

inline Variable pow(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Power>({a, b})[0];
}

inline Variable eq(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Equal>({a, b})[0];
}

inline Variable ne(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<NotEqual>({a, b})[0];
}

inline Variable ge(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<GreaterEqual>({a, b})[0];
}

inline Variable le(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<LessEqual>({a, b})[0];
}

inline Variable gt(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Greater>({a, b})[0];
}

inline Variable lt(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Less>({a, b})[0];
}

}  // namespace lmp::autograd::ops
