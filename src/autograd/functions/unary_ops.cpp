#include "include/lamppp/autograd/functions/unary_ops.hpp"
#include <cassert>
#include <cmath>
#include "include/lamppp/autograd/functions/overloads.hpp"
#include "include/lamppp/autograd/variable.hpp"
#include "include/lamppp/tensor/tensor.hpp"

namespace lmp::autograd::ops {

variable_list ExponentialBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  Variable exp_var(
      tensor::ops::exp(self.data()));  // TODO(nlin): these can all be cached!!!
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

  Variable recip_var(1.0 / self);
  self.incr_grad(recip_var.data() * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list ReLUBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  Variable relu_var(self > 0.0);
  self.incr_grad(relu_var.data() * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Exponential::execute(const variable_list& inputs) {
  assert(inputs.size() == 1 && "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::exp(self.data());
}

tensor::Tensor Logarithm::execute(const variable_list& inputs) {
  assert(inputs.size() == 1 && "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::log(self.data());
}

tensor::Tensor ReLU::execute(const variable_list& inputs) {
  assert(inputs.size() == 1 && "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::relu(self.data());
}

}  // namespace lmp::autograd::ops