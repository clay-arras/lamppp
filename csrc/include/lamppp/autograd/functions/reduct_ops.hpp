#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"

namespace lmp::autograd::ops {

/// @internal
struct SummationBackward : public Function {
  size_t axis_;
  explicit SummationBackward(size_t axis) : axis_(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct MaximumBackward : public Function {
  size_t axis_;
  explicit MaximumBackward(size_t axis) : axis_(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct MinimumBackward : public Function {
  size_t axis_;
  explicit MinimumBackward(size_t axis) : axis_(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct ProductBackward : public Function {
  size_t axis_;
  explicit ProductBackward(size_t axis) : axis_(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Summation : public ForwardFunction<Summation> {
  using DefaultBackward = SummationBackward;
  size_t axis_;
  explicit Summation(size_t axis) : axis_(axis) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Maximum : public ForwardFunction<Maximum> {
  using DefaultBackward = MaximumBackward;
  size_t axis_;
  explicit Maximum(size_t axis) : axis_(axis) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Minimum : public ForwardFunction<Minimum> {
  using DefaultBackward = MinimumBackward;
  size_t axis_;
  explicit Minimum(size_t axis) : axis_(axis) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct Product : public ForwardFunction<Product> {
  using DefaultBackward = ProductBackward;
  size_t axis_;
  explicit Product(size_t axis) : axis_(axis) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};
/// @endinternal

/**
 * @brief Sum a variable along an axis
 * @param a The variable to sum
 * @param axis The axis to sum along
 * @return The result of the summation
 */
inline Variable sum(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Summation>({a}, axis)[0];
}

/**
 * @brief Maximum a variable along an axis
 * @param a The variable to maximum
 * @param axis The axis to maximum along
 * @return The result of the maximum
 */
inline Variable max(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Maximum>({a}, axis)[0];
}

/**
 * @brief Minimum a variable along an axis
 * @param a The variable to minimum
 * @param axis The axis to minimum along
 * @return The result of the minimum
 */
inline Variable min(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Minimum>({a}, axis)[0];
}

/**
 * @brief Product a variable along an axis
 * @param a The variable to product
 * @param axis The axis to product along
 * @return The result of the product
 */
inline Variable prod(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Product>({a}, axis)[0];
}

}  // namespace lmp::autograd::ops
