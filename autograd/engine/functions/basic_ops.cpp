#include "basic_ops.hpp"
#include <cassert>
#include <memory>
#include "autograd/engine/function.hpp"
#include "autograd/engine/variable.hpp"

namespace autograd {

variable_list AddBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(grad.grad());
  other.incr_grad(grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list SubtractBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(grad.grad());
  other.incr_grad((-1.0) * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list MultiplyBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(other.data() * grad.grad());
  other.incr_grad(self.data() * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list DivideBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(grad.grad() / other.data());
  other.incr_grad((-1.0) *
                  (self.data() * grad.grad() / (other.data() * other.data())));

  variable_list grad_inputs = {};
  return grad_inputs;
}

// TODO(nlin): need to optimize s.t. if requires_grad is false then it doesn't do the make_shared
Tensor Add::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() + other.data();
}

Tensor Subtract::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() - other.data();
}

Tensor Multiply::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() * other.data();
}

Tensor Divide::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() / other.data();
}

}  // namespace autograd