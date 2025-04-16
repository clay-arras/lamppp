#include "basic_ops.h"
#include <cassert>
#include "matrix_ops.h"
#include "autograd/engine/function.h"
#include "autograd/engine/variable.h"

namespace autograd {

variable_list MatrixMultiplicationBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(grad.grad() * other.data().T());
  other.incr_grad(self.data().T() * grad.grad());

  variable_list grad_inputs = {grad, grad}; // TODO(nlin): remove these maybe, this isn't right
  return grad_inputs;
}

variable_list MatrixMultiplication::apply(const variable_list& inputs) {
  assert(inputs.size() == 2);
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  Variable result = Variable(self.data().matmul(other.data()), true);
  auto backward_fn = std::make_shared<MatrixMultiplicationBackward>();
  backward_fn->saved_inputs =
      std::make_unique<variable_list>(variable_list{self, other});
  result.set_grad_fn(backward_fn);
  return {result};
}

}