#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"

namespace lmp::autograd::ops {

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

struct MinimumBackward : public Function {
  size_t axis;
  explicit MinimumBackward(size_t axis) : axis(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct ProductBackward : public Function {
  size_t axis;
  explicit ProductBackward(size_t axis) : axis(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Summation : public ForwardFunction<Summation> {
  using DefaultBackward = SummationBackward;
  size_t axis;
  explicit Summation(size_t axis) : axis(axis) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Maximum : public ForwardFunction<Maximum> {
  using DefaultBackward = MaximumBackward;
  size_t axis;
  explicit Maximum(size_t axis) : axis(axis) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Minimum : public ForwardFunction<Minimum> {
  using DefaultBackward = MinimumBackward;
  size_t axis;
  explicit Minimum(size_t axis) : axis(axis) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Product : public ForwardFunction<Product> {
  using DefaultBackward = ProductBackward;
  size_t axis;
  explicit Product(size_t axis) : axis(axis) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};

inline Variable sum(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Summation>({a}, axis)[0];
}

inline Variable max(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Maximum>({a}, axis)[0];
}

inline Variable min(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Minimum>({a}, axis)[0];
}

inline Variable prod(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Product>({a}, axis)[0];
}

}  // namespace lmp::autograd::ops
