#include "lamppp/autograd/functions/basic_ops.hpp"
#include <cassert>
#include <memory>
#include "lamppp/autograd/function.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/tensor/functions/reduct_ops.hpp"

namespace lmp::autograd::ops {

variable_list AddBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  tensor::Tensor self_grad = grad.grad();
  tensor::Tensor other_grad = grad.grad();

  // TODO: this code is ugly and can be improved
  size_t out_dims = grad.grad().shape().size();
#pragma unroll
  for (size_t i = LMP_MAX_DIMS; i-- > 0;) {
    if (i >= out_dims)
      continue;

    int offset = out_dims - 1 - i;
    int a_idx = self.data().shape().size() - 1 - offset;
    int b_idx = other.data().shape().size() - 1 - offset;

    int a_val = (a_idx >= 0 ? self.data().shape()[a_idx] : -1);
    int b_val = (b_idx >= 0 ? other.data().shape()[b_idx] : -1);

    if (a_val == 1 || a_val == -1) {
      self_grad = tensor::ops::sum(self_grad, i);
    }
    if (b_val == 1 || b_val == -1) {
      other_grad = tensor::ops::sum(other_grad, i);
    }

    while (self_grad.shape().size() > self.data().shape().size()) {
      self_grad = tensor::ops::sum(self_grad, 0);
      self_grad.squeeze(0);
    }
    while (other_grad.shape().size() > other.data().shape().size()) {
      other_grad = tensor::ops::sum(other_grad, 0);
      other_grad.squeeze(0);
    }
  }

  self.incr_grad(self_grad);
  other.incr_grad(other_grad);

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list SubtractBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  tensor::Tensor self_grad = grad.grad();
  tensor::Tensor other_grad = (-1.0) * grad.grad();

  size_t out_dims = grad.grad().shape().size();
#pragma unroll
  for (size_t i = LMP_MAX_DIMS; i-- > 0;) {
    if (i >= out_dims)
      continue;

    int offset = out_dims - 1 - i;
    int a_idx = self.data().shape().size() - 1 - offset;
    int b_idx = other.data().shape().size() - 1 - offset;

    int a_val = (a_idx >= 0 ? self.data().shape()[a_idx] : -1);
    int b_val = (b_idx >= 0 ? other.data().shape()[b_idx] : -1);

    if (a_val == 1 || a_val == -1) {
      self_grad = tensor::ops::sum(self_grad, i);
    }
    if (b_val == 1 || b_val == -1) {
      other_grad = tensor::ops::sum(other_grad, i);
    }

    while (self_grad.shape().size() > self.data().shape().size()) {
      self_grad = tensor::ops::sum(self_grad, 0);
      self_grad.squeeze(0);
    }
    while (other_grad.shape().size() > other.data().shape().size()) {
      other_grad = tensor::ops::sum(other_grad, 0);
      other_grad.squeeze(0);
    }
  }

  self.incr_grad(self_grad);
  other.incr_grad(other_grad);

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list MultiplyBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  tensor::Tensor self_grad = other.data() * grad.grad();
  tensor::Tensor other_grad = self.data() * grad.grad();

  size_t out_dims = grad.grad().shape().size();
#pragma unroll
  for (size_t i = LMP_MAX_DIMS; i-- > 0;) {
    if (i >= out_dims)
      continue;

    int offset = out_dims - 1 - i;
    int a_idx = self.data().shape().size() - 1 - offset;
    int b_idx = other.data().shape().size() - 1 - offset;

    int a_val = (a_idx >= 0 ? self.data().shape()[a_idx] : -1);
    int b_val = (b_idx >= 0 ? other.data().shape()[b_idx] : -1);

    if (a_val == 1 || a_val == -1) {
      self_grad = tensor::ops::sum(self_grad, i);
    }
    if (b_val == 1 || b_val == -1) {
      other_grad = tensor::ops::sum(other_grad, i);
    }

    while (self_grad.shape().size() > self.data().shape().size()) {
      self_grad = tensor::ops::sum(self_grad, 0);
      self_grad.squeeze(0);
    }
    while (other_grad.shape().size() > other.data().shape().size()) {
      other_grad = tensor::ops::sum(other_grad, 0);
      other_grad.squeeze(0);
    }
  }

  self.incr_grad(self_grad);
  other.incr_grad(other_grad);

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list DivideBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  tensor::Tensor self_grad = grad.grad() / other.data();
  tensor::Tensor other_grad =
      (-1.0) * (self.data() * grad.grad() / (other.data() * other.data()));

  size_t out_dims = grad.grad().shape().size();
#pragma unroll
  for (size_t i = LMP_MAX_DIMS; i-- > 0;) {
    if (i >= out_dims)
      continue;

    int offset = out_dims - 1 - i;
    int a_idx = self.data().shape().size() - 1 - offset;
    int b_idx = other.data().shape().size() - 1 - offset;

    int a_val = (a_idx >= 0 ? self.data().shape()[a_idx] : -1);
    int b_val = (b_idx >= 0 ? other.data().shape()[b_idx] : -1);

    if (a_val == 1 || a_val == -1) {
      self_grad = tensor::ops::sum(self_grad, i);
    }
    if (b_val == 1 || b_val == -1) {
      other_grad = tensor::ops::sum(other_grad, i);
    }

    while (self_grad.shape().size() > self.data().shape().size()) {
      self_grad = tensor::ops::sum(self_grad, 0);
      self_grad.squeeze(0);
    }
    while (other_grad.shape().size() > other.data().shape().size()) {
      other_grad = tensor::ops::sum(other_grad, 0);
      other_grad.squeeze(0);
    }
  }

  self.incr_grad(self_grad);
  other.incr_grad(other_grad);

  variable_list grad_inputs = {};
  return grad_inputs;
}

// TODO(nlin): need to optimize s.t. if requires_grad is false then it doesn't do the make_shared
tensor::Tensor Add::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() + other.data();
}

tensor::Tensor Subtract::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() - other.data();
}

tensor::Tensor Multiply::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() * other.data();
}

tensor::Tensor Divide::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return self.data() / other.data();
}

}  // namespace lmp::autograd::ops