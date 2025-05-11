#include "include/lamppp/autograd/functions/binary_ops.hpp"
#include <cassert>
#include "include/lamppp/autograd/variable.hpp"
#include "include/lamppp/tensor/tensor.hpp"

namespace autograd {

variable_list EqualBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list LessBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list LessEqualBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list NotEqualBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list GreaterBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list GreaterEqualBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& lhs = (*saved_inputs)[0];
  Variable& rhs = (*saved_inputs)[1];

  variable_list grad_inputs = {};
  return grad_inputs;
}

Tensor Equal::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() == rhs.data();
}

Tensor Less::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() < rhs.data();
}

Tensor LessEqual::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() <= rhs.data();
}

Tensor NotEqual::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() != rhs.data();
}

Tensor Greater::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() > rhs.data();
}

Tensor GreaterEqual::execute(const variable_list& inputs) {
  assert(inputs.size() == 2 && "Function must take 2 inputs");
  const Variable& lhs = inputs[0];
  const Variable& rhs = inputs[1];

  return lhs.data() >= rhs.data();
}

}  // namespace autograd
