#include "unary_ops.h"
#include <cassert>
#include <cmath>
#include "autograd/engine/tensor.h"
#include "autograd/engine/variable.h"
#include "autograd/engine/variable_ops.h"

namespace autograd {

variable_list ExponentialBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  Variable exp_var(
      self.data().exp());  // TODO(nlin): these can all be cached!!!
  self.incr_grad(
      exp_var.data() *
      grad.grad());  // TODO(nlin): maybe will this result in recursion? higher order derivatives

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list LogarithmBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  Variable recip_var(1 / self);
  self.incr_grad(recip_var.data() * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list ReLUBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  Variable relu_var(self > 0.0F);
  self.incr_grad(relu_var.data() * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

Tensor Exponential::execute(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  return self.data().exp();
}

Tensor Logarithm::execute(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  return self.data().log();
}

Tensor ReLU::execute(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  return self.data().relu();
}

}  // namespace autograd