#include "lamppp/autograd/functions/view_ops.hpp"
#include <cassert>
#include <cmath>
#include "lamppp/autograd/variable.hpp"
#include "lamppp/tensor/fill_like.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::autograd::ops {

variable_list ReshapeBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(grad.grad().reshape(self.data().shape()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Reshape::execute(const variable_list& inputs) {
  assert(inputs.size() == 1 && "Function must take one inputs");
  const Variable& self = inputs[0];

  return self.data().reshape(shape);
}

variable_list SqueezeBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(grad.grad().expand_dims(axis));

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Squeeze::execute(const variable_list& inputs) {
  assert(inputs.size() == 1 && "Function must take one input");
  const Variable& self = inputs[0];

  return self.data().squeeze(axis);
}

variable_list ExpandDimsBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(grad.grad().squeeze(axis));

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor ExpandDims::execute(const variable_list& inputs) {
  assert(inputs.size() == 1 && "Function must take one inputs");
  const Variable& self = inputs[0];

  return self.data().expand_dims(axis);
}

}  // namespace lmp::autograd::ops