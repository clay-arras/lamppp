#pragma once

#include <include/lamppp/autograd/forward_function.hpp>
#include <include/lamppp/autograd/function.hpp>

namespace autograd {

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
  static Tensor execute(const variable_list& inputs);
};

struct Subtract : public ForwardFunction<Subtract> {
  using DefaultBackward = SubtractBackward;
  static Tensor execute(const variable_list& inputs);
};

struct Multiply : public ForwardFunction<Multiply> {
  using DefaultBackward = MultiplyBackward;
  static Tensor execute(const variable_list& inputs);
};

struct Divide : public ForwardFunction<Divide> {
  using DefaultBackward = DivideBackward;
  static Tensor execute(const variable_list& inputs);
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

}  // namespace autograd
