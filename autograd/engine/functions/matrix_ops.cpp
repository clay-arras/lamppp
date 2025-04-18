#include "matrix_ops.h"
#include <cassert>
#include "autograd/engine/function.h"
#include "autograd/engine/variable.h"

namespace autograd {

variable_list MatrixMultiplicationBackward::apply(
    const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(grad.grad().matmul(other.data().transpose()));
  other.incr_grad(self.data().transpose().matmul(grad.grad()));

  variable_list grad_inputs =
      {};  // TODO(nlin): remove these maybe, this isn't right
  return grad_inputs;
}

variable_list TransposeBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(grad.grad().transpose());

  variable_list grad_inputs = {};
  return grad_inputs;
}

Tensor MatrixMultiplication::execute(const variable_list& inputs) {
  assert(inputs.size() == 2);
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data().matmul(other.data());
}

Tensor Transpose::execute(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  return self.data().transpose();
}

}  // namespace autograd