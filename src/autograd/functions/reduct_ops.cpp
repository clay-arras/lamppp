#include "lamppp/autograd/functions/reduct_ops.hpp"

#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/fill_like.hpp"

namespace lmp::autograd::ops {

variable_list SummationBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(tensor::ones_like(self.data()) * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list MaximumBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  tensor::Tensor mask = tensor::ops::equal(self.data(), grad.data());
  self.incr_grad(grad.grad() * mask);

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list MinimumBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  tensor::Tensor mask = tensor::ops::equal(self.data(), grad.data());
  self.incr_grad(grad.grad() * mask);

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Summation::execute(const variable_list& inputs) const {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self_var = inputs[0];
  return tensor::ops::sum(self_var.data(), axis);
}

tensor::Tensor Maximum::execute(const variable_list& inputs) const {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self_var = inputs[0];
  return tensor::ops::max(self_var.data(), axis);
}

tensor::Tensor Minimum::execute(const variable_list& inputs) const {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self_var = inputs[0];
  return tensor::ops::min(self_var.data(), axis);
}

}  // namespace lmp::autograd::ops
