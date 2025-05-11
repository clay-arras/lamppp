#pragma once

#include "include/lamppp/autograd/forward_function.hpp"
#include "include/lamppp/autograd/function.hpp"

namespace autograd {

struct MatrixMultiplicationBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct TransposeBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct MatrixMultiplication : public ForwardFunction<MatrixMultiplication> {
  using DefaultBackward = MatrixMultiplicationBackward;
  static Tensor execute(const variable_list& inputs);
};

struct Transpose : public ForwardFunction<Transpose> {
  using DefaultBackward = TransposeBackward;
  static Tensor execute(const variable_list& inputs);
};

}  // namespace autograd
