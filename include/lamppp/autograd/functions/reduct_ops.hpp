#pragma once

#include "include/lamppp/autograd/forward_function.hpp"
#include "include/lamppp/autograd/function.hpp"
#include "include/lamppp/autograd/functions/overloads.hpp"

namespace autograd {

struct SummationBackward : public Function {
  size_t axis;
  explicit SummationBackward(size_t axis) : axis(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct MaximumBackward : public Function {
  size_t axis;
  explicit MaximumBackward(size_t axis) : axis(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Summation : public ForwardFunction<Summation> {
  using DefaultBackward = SummationBackward;
  size_t axis;
  explicit Summation(size_t axis) : axis(axis) {}
  Tensor execute(const variable_list& inputs) const;
};

struct Maximum : public ForwardFunction<Maximum> {
  using DefaultBackward = MaximumBackward;
  size_t axis;
  explicit Maximum(size_t axis) : axis(axis) {}
  Tensor execute(const variable_list& inputs) const;
};

inline Variable sum(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Summation>({a}, axis)[0];
}

inline Variable max(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Maximum>({a}, axis)[0];
}

}  // namespace autograd
