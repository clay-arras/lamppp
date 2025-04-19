#pragma once

#ifndef _MATRIX_OPS_H_
#define _MATRIX_OPS_H_

#include "autograd/engine/forward_function.h"
#include "autograd/engine/function.h"

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

#endif  // _MATRIX_OPS_H_