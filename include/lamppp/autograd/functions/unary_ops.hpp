#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"

namespace lmp::autograd::ops {

/// @internal
struct NegationBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Negation : public ForwardFunction<Negation> {
  using DefaultBackward = NegationBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct ExponentialBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Exponential : public ForwardFunction<Exponential> {
  using DefaultBackward = ExponentialBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct LogarithmBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Logarithm : public ForwardFunction<Logarithm> {
  using DefaultBackward = LogarithmBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct SquareRootBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct SquareRoot : public ForwardFunction<SquareRoot> {
  using DefaultBackward = SquareRootBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct AbsoluteValueBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct AbsoluteValue : public ForwardFunction<AbsoluteValue> {
  using DefaultBackward = AbsoluteValueBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct SineBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Sine : public ForwardFunction<Sine> {
  using DefaultBackward = SineBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct CosineBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Cosine : public ForwardFunction<Cosine> {
  using DefaultBackward = CosineBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct TangentBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Tangent : public ForwardFunction<Tangent> {
  using DefaultBackward = TangentBackward;
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct ClampBackward : public Function {
  tensor::Scalar min_val_, max_val_;
  explicit ClampBackward(tensor::Scalar min_val, tensor::Scalar max_val)
      : min_val_(min_val), max_val_(max_val) {}
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Clamp : public ForwardFunction<Clamp> {
  using DefaultBackward = ClampBackward;
  tensor::Scalar min_val_, max_val_;
  explicit Clamp(tensor::Scalar min_val, tensor::Scalar max_val)
      : min_val_(min_val), max_val_(max_val) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};
/// @endinternal

/**
 * @brief Negate a variable
 * @param a The variable to negate
 * @return The result of the negation
 */
inline Variable neg(const Variable& a) {
  return VariableOpFact::apply<Negation>({a})[0];
}

/**
 * @brief Exponentiate a variable
 * @param a The variable to exponentiate
 * @return The result of the exponentiation
 */
inline Variable exp(const Variable& a) {
  return VariableOpFact::apply<Exponential>({a})[0];
}

/**
 * @brief Take the logarithm of a variable
 * @param a The variable to take the logarithm of
 * @return The result of the logarithm
 */
inline Variable log(const Variable& a) {
  return VariableOpFact::apply<Logarithm>({a})[0];
}

/**
 * @brief Take the square root of a variable
 * @param a The variable to take the square root of
 * @return The result of the square root
 */
inline Variable sqrt(const Variable& a) {
  return VariableOpFact::apply<SquareRoot>({a})[0];
}

/**
 * @brief Take the absolute value of a variable
 * @param a The variable to take the absolute value of
 * @return The result of the absolute value
 */
inline Variable abs(const Variable& a) {
  return VariableOpFact::apply<AbsoluteValue>({a})[0];
}

/**
 * @brief Take the sine of a variable
 * @param a The variable to take the sine of
 * @return The result of the sine
 */
inline Variable sin(const Variable& a) {
  return VariableOpFact::apply<Sine>({a})[0];
}

/**
 * @brief Take the cosine of a variable
 * @param a The variable to take the cosine of
 * @return The result of the cosine
 */
inline Variable cos(const Variable& a) {
  return VariableOpFact::apply<Cosine>({a})[0];
}

/**
 * @brief Take the tangent of a variable
 * @param a The variable to take the tangent of
 * @return The result of the tangent
 */
inline Variable tan(const Variable& a) {
  return VariableOpFact::apply<Tangent>({a})[0];
}

/**
 * @brief Clamp a variable between a minimum and maximum value
 * @param a The variable to clamp
 * @param min_val The minimum value
 * @param max_val The maximum value
 * @return The result of the clamping
 */
inline Variable clamp(const Variable& a, tensor::Scalar min_val,
                      tensor::Scalar max_val) {
  return VariableOpFact::apply<Clamp>({a}, min_val, max_val)[0];
}

}  // namespace lmp::autograd::ops