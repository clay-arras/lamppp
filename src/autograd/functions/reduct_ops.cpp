#include "include/lamppp/autograd/functions/reduct_ops.hpp"
#include <cassert>
#include "include/lamppp/autograd/variable.hpp"
#include "include/lamppp/tensor/tensor_helper.hpp"

namespace lmp::autograd::ops {

variable_list SummationBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  self.incr_grad(tensor::ones_like(self.data()) * grad.grad());

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list MaximumBackward::apply(
    const variable_list&
        gradOutputs) {  // TODO(nlin): this is disgusting + slow, values can be cached

  variable_list grad_inputs = {};
  return grad_inputs;
}

tensor::Tensor Summation::execute(const variable_list& inputs) const {
  assert(inputs.size() == 1 && "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::sum(self.data(), axis);
}

tensor::Tensor Maximum::execute(const variable_list& inputs) const {
  assert(inputs.size() == 1 && "Function must take one input");
  const Variable& self = inputs[0];

  return tensor::ops::max(self.data(), axis);
}

}  // namespace lmp::autograd::ops
