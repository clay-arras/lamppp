#include "lamppp/autograd/functions/unary_ops.hpp"

#include <cmath>
#include "lamppp/autograd/functions/overloads.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/fill_like.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::autograd::ops {

variable_list ExponentialBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(grad.data() * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Exponential::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::exp(self.data());
}

variable_list LogarithmBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad((1 / self.data()) * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Logarithm::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::log(self.data());
}

variable_list SqrtBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  Variable sqrt_var(1 / (2 * grad.data()));
  self.incr_grad(sqrt_var.data() * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Sqrt::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::sqrt(self.data());
}

variable_list AbsBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  Variable sign_self(ones_like(self.data()) * (self.data() > 0.0) -
                     (self.data() < 0.0));
  self.incr_grad(sign_self.data() * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Abs::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::abs(self.data());
}

variable_list SinBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(grad.grad() * tensor::ops::cos(self.data()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Sin::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::sin(self.data());
}

variable_list CosBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(-1 * tensor::ops::sin(self.data()) * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Cos::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::cos(self.data());
}

variable_list TanBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  Variable cos_var(tensor::ops::cos(self.data()));
  self.incr_grad((1.0 / (cos_var.data() * cos_var.data())) * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Tan::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::tan(self.data());
}

variable_list ClampBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  tensor::Tensor mask =
      (tensor::ones_like(self.data()) * (self.data() > min_val_)) *
      (self.data() < max_val_);
  self.incr_grad(mask * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Clamp::execute(const variable_list& inputs) const {
  LMP_INTERNAL_ASSERT(inputs.size() == 1, "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::clamp(self.data(), min_val_, max_val_);
}

}  // namespace lmp::autograd::ops