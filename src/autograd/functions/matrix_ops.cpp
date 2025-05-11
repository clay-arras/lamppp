#include "include/lamppp/autograd/functions/matrix_ops.hpp"
#include <cassert>
#include "include/lamppp/autograd/function.hpp"
#include "include/lamppp/autograd/variable.hpp"

namespace autograd {

variable_list MatrixMultiplicationBackward::apply(
    const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(matmul(grad.grad(), transpose(other.data())));
  other.incr_grad(matmul(transpose(self.data()), grad.grad()));

  variable_list grad_inputs =
      {};  // TODO(nlin): remove these maybe, this isn't right
  return grad_inputs;
}

variable_list TransposeBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(transpose(grad.grad()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

Tensor MatrixMultiplication::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return matmul(self.data(), other.data());
}

Tensor Transpose::execute(const variable_list& inputs) {
  assert(inputs.size() == 1 && "Function must take one input");
  const Variable& self = inputs[0];

  return transpose(self.data());
}

}  // namespace autograd