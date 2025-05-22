#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"

namespace lmp::autograd::ops {

struct ExponentialBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Exponential : public ForwardFunction<Exponential> {
  using DefaultBackward = ExponentialBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct LogarithmBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Logarithm : public ForwardFunction<Logarithm> {
  using DefaultBackward = LogarithmBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct SqrtBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Sqrt : public ForwardFunction<Sqrt> {
  using DefaultBackward = SqrtBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct AbsBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Abs : public ForwardFunction<Abs> {
  using DefaultBackward = AbsBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct SinBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Sin : public ForwardFunction<Sin> {
  using DefaultBackward = SinBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct CosBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Cos : public ForwardFunction<Cos> {
  using DefaultBackward = CosBackward;
  static tensor::Tensor execute(const variable_list& inputs);
};

struct TanBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Tan : public ForwardFunction<Tan> {
  using DefaultBackward = TanBackward;
  static tensor::Tensor execute(const variable_list& inputs);
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

inline Variable exp(const Variable& a) {
  return VariableOpFact::apply<Exponential>({a})[0];
}

inline Variable log(const Variable& a) {
  return VariableOpFact::apply<Logarithm>({a})[0];
}

inline Variable sqrt(const Variable& a) {
  return VariableOpFact::apply<Sqrt>({a})[0];
}

inline Variable abs(const Variable& a) {
  return VariableOpFact::apply<Abs>({a})[0];
}

inline Variable sin(const Variable& a) {
  return VariableOpFact::apply<Sin>({a})[0];
}

inline Variable cos(const Variable& a) {
  return VariableOpFact::apply<Cos>({a})[0];
}

inline Variable tan(const Variable& a) {
  return VariableOpFact::apply<Tan>({a})[0];
}

inline Variable clamp(const Variable& a, tensor::Scalar min_val,
                      tensor::Scalar max_val) {
  return VariableOpFact::apply<Clamp>({a}, min_val, max_val)[0];
}

}  // namespace lmp::autograd::ops