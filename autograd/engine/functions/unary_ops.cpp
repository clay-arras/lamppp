#include "unary_ops.h"
#include <cassert>
#include <cmath>
#include "autograd/engine/variable.h"
#include "autograd/engine/variable_ops.h"

namespace autograd {

variable_list ExponentialBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  
  Variable exp_var(self.data().exp());
  self.incr_grad(exp_var.data() * grad.grad()); // TODO(nlin): maybe will this result in recursion? higher order derivatives

  variable_list grad_inputs = {exp_var * grad}; 
  return grad_inputs;
}

variable_list LogarithmBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  
  Variable recip_var(1 / self);
  self.incr_grad(recip_var.data() * grad.grad());

  variable_list grad_inputs = {recip_var * grad}; 
  return grad_inputs;
}

variable_list ReLUBackward::apply(const variable_list& gradOutputs) {
  assert(0); // NOTE: not implemented for now, PLEASE ADD MASKING WITH EIGEN S.T == returns a tensor mask
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  
  Tensor mask(self.data().data, self.data().shape);
  for (auto& val : mask.data) {
    val = (val > 0) ? 1.0F : 0.0F;
  }
  Variable mask_var(mask);
  variable_list grad_inputs = {grad * mask_var}; 
  return grad_inputs;
}

variable_list Exponential::apply(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  Variable result = Variable(self.data().exp(), self.requires_grad());
  auto backward_fn = std::make_shared<ExponentialBackward>();
  backward_fn->saved_inputs = std::make_unique<variable_list>(variable_list{self});
  result.set_grad_fn(backward_fn);
  
  return {result};
}

variable_list Logarithm::apply(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  Variable result = Variable(self.data().log(), self.requires_grad());
  auto backward_fn = std::make_shared<LogarithmBackward>();
  backward_fn->saved_inputs = std::make_unique<variable_list>(variable_list{self});
  result.set_grad_fn(backward_fn);
  
  return {result};
}

variable_list ReLU::apply(const variable_list& inputs) {
  assert(0); // TODO(nlin): NOT IMPLEMENTED
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  Variable result = Variable(self.data().relu(), self.requires_grad());
  
  if (self.requires_grad()) {
    auto backward_fn = std::make_shared<ReLUBackward>();
    backward_fn->saved_inputs = std::make_unique<variable_list>(variable_list{self});
    result.set_grad_fn(backward_fn);
  }
  
  return {result};
}

}