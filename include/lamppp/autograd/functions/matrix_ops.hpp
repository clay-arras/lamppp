#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"

namespace lmp::autograd::ops {

struct MatrixMultiplicationBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct TransposeBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct MatrixMultiplication : public ForwardFunction<MatrixMultiplication> {
  using DefaultBackward = MatrixMultiplicationBackward;
  tensor::Tensor execute(const variable_list& inputs);
};

struct Transpose : public ForwardFunction<Transpose> {
  using DefaultBackward = TransposeBackward;
  tensor::Tensor execute(const variable_list& inputs);
};

inline Variable matmul(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<MatrixMultiplication>({a, b})[0];
}

inline Variable transpose(const Variable& a) {
  return VariableOpFact::apply<Transpose>({a})[0];
}

}  // namespace lmp::autograd::ops
