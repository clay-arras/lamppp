#include "lamppp/autograd/functions/expand_ops.hpp"
#include <memory>
#include "lamppp/autograd/function.hpp"
#include "lamppp/autograd/grad_utils.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/assert.hpp"

namespace lmp::autograd::ops {

variable_list AddBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(detail::sum_broadcast_axis(grad.grad(), self.data().shape()));
  other.incr_grad(
      detail::sum_broadcast_axis(grad.grad(), other.data().shape()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list SubtractBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(detail::sum_broadcast_axis(grad.grad(), self.data().shape()));
  other.incr_grad(
      detail::sum_broadcast_axis((-1) * grad.grad(), other.data().shape()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list MultiplyBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(detail::sum_broadcast_axis(other.data() * grad.grad(),
                                            self.data().shape()));
  other.incr_grad(detail::sum_broadcast_axis(self.data() * grad.grad(),
                                             other.data().shape()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list DivideBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  tensor::Tensor other_grad =
      (-1.0) * (grad.data() * grad.grad() / (other.data()));

  self.incr_grad(detail::sum_broadcast_axis(grad.grad() / other.data(),
                                            self.data().shape()));
  other.incr_grad(detail::sum_broadcast_axis(other_grad, other.data().shape()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list PowerBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  tensor::Tensor self_grad = grad.grad() * other.data() * tensor::ops::pow(self.data(), other.data() - 1);
  tensor::Tensor other_grad = grad.grad() * grad.data() * tensor::ops::log(self.data());

  self.incr_grad(detail::sum_broadcast_axis(self_grad, self.data().shape()));
  other.incr_grad(detail::sum_broadcast_axis(other_grad, other.data().shape()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

// TODO(nlin): need to optimize s.t. if requires_grad is false then it doesn't do the make_shared
tensor::Tensor Add::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() + other.data();
}

tensor::Tensor Subtract::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() - other.data();
}

tensor::Tensor Multiply::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() * other.data();
}

tensor::Tensor Divide::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() / other.data();
}

tensor::Tensor Power::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return tensor::ops::pow(self.data(), other.data());
}

variable_list EqualBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list LessBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list LessEqualBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list NotEqualBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list GreaterBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list GreaterEqualBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1, "Output size mismatch.");
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Equal::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() == rhs.data();
}

tensor::Tensor Less::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() < rhs.data();
}

tensor::Tensor LessEqual::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() <= rhs.data();
}

tensor::Tensor NotEqual::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() != rhs.data();
}

tensor::Tensor Greater::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() > rhs.data();
}

tensor::Tensor GreaterEqual::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2, "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() >= rhs.data();
}

}  // namespace lmp::autograd::ops