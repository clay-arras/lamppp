#include "lamppp/autograd/functions/matrix_ops.hpp"
#include "lamppp/autograd/function.hpp"
#include "lamppp/autograd/variable.hpp"

namespace lmp::autograd::ops {

variable_list MatrixMultiplicationBackward::apply(
    const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch.";
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  self.incr_grad(
      tensor::ops::matmul(grad.grad(), tensor::ops::transpose(other.data())));
  other.incr_grad(
      tensor::ops::matmul(tensor::ops::transpose(self.data()), grad.grad()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list TransposeBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch.";
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(tensor::ops::transpose(grad.grad()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor MatrixMultiplication::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2) << "Function must take 2 inputs";
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return tensor::ops::matmul(self.data(), other.data());
}

tensor::Tensor Transpose::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 1) << "Function must take one input";
  const Variable& self = inputs[0];

  return tensor::ops::transpose(self.data());
}

}  // namespace lmp::autograd::ops