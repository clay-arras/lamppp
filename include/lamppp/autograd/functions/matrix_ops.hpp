#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"

namespace lmp::autograd::ops {

/// @internal
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
/// @endinternal

/**
 * @brief Matrix multiplication between two variables
 * @param a The first variable
 * @param b The second variable
 * @return The result of the matrix multiplication
 */
inline Variable matmul(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<MatrixMultiplication>({a, b})[0];
}

/**
 * @brief Transpose a variable
 * @param a The variable to transpose
 * @return The transposed variable
 * @note this function creates a new variable, not a view.
 */
inline Variable transpose(const Variable& a) {
  return VariableOpFact::apply<Transpose>({a})[0];
}

}  // namespace lmp::autograd::ops
