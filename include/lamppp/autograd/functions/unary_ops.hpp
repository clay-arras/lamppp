#pragma once

#include "include/lamppp/autograd/forward_function.hpp"
#include "include/lamppp/autograd/function.hpp"
#include "include/lamppp/autograd/functions/overloads.hpp"

namespace autograd {

struct ExponentialBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct LogarithmBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct ReLUBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Exponential : public ForwardFunction<Exponential> {
  using DefaultBackward = ExponentialBackward;
  static Tensor execute(const variable_list& inputs);
};

struct Logarithm : public ForwardFunction<Logarithm> {
  using DefaultBackward = LogarithmBackward;
  static Tensor execute(const variable_list& inputs);
};

struct ReLU : public ForwardFunction<ReLU> {
  using DefaultBackward = ReLUBackward;
  static Tensor execute(const variable_list& inputs);
};

inline Variable exp(const Variable& a) {
  return VariableOpFact::apply<Exponential>({a})[0];
}

inline Variable log(const Variable& a) {
  return VariableOpFact::apply<Logarithm>({a})[0];
}

inline Variable relu(const Variable& a) {
  return VariableOpFact::apply<ReLU>({a})[0];
}

}  // namespace autograd