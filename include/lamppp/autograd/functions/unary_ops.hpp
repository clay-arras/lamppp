#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"

namespace lmp::autograd::ops {

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

inline Variable neg(const Variable& a) {
  return VariableOpFact::apply<Negation>({a})[0];
}

inline Variable exp(const Variable& a) {
  return VariableOpFact::apply<Exponential>({a})[0];
}

inline Variable log(const Variable& a) {
  return VariableOpFact::apply<Logarithm>({a})[0];
}

inline Variable sqrt(const Variable& a) {
  return VariableOpFact::apply<SquareRoot>({a})[0];
}

inline Variable abs(const Variable& a) {
  return VariableOpFact::apply<AbsoluteValue>({a})[0];
}

inline Variable sin(const Variable& a) {
  return VariableOpFact::apply<Sine>({a})[0];
}

inline Variable cos(const Variable& a) {
  return VariableOpFact::apply<Cosine>({a})[0];
}

inline Variable tan(const Variable& a) {
  return VariableOpFact::apply<Tangent>({a})[0];
}

inline Variable clamp(const Variable& a, tensor::Scalar min_val,
                      tensor::Scalar max_val) {
  return VariableOpFact::apply<Clamp>({a}, min_val, max_val)[0];
}

}  // namespace lmp::autograd::ops