#include "unary_ops.hpp"
#include <cassert>
#include <cmath>
#include "autograd/engine/tensor.hpp"
#include "autograd/engine/variable.hpp"
#include "autograd/engine/variable_ops.hpp"

namespace autograd {

variable_list ExponentialBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  Variable exp_var(exp(self.data()));  // TODO(nlin): these can all be cached!!!
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

  // std::cout << "Before Self Var: " << self << std::endl;
  // std::cout << "Befoore Test Var: " << (self > 0.0) << std::endl;
  // std::cout << "Afterj test Self Var: " << self << std::endl;

  Variable relu_var(self > 0.0F);
  // std::cout << "Relu Var: " << relu_var << std::endl;
  // std::cout << "Afterj Self Var: " << self << std::endl;
  // std::cout << "Afterj Test Var: " << (self > 0.0) << std::endl;

  self.incr_grad(relu_var.data() * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

Tensor Exponential::execute(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  return exp(self.data());
}

Tensor Logarithm::execute(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  return log(self.data());
}

Tensor ReLU::execute(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  return relu(self.data());
}

}  // namespace autograd