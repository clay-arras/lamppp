#pragma once

#include "lamppp/autograd/functions/reduct_ops.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/assert.hpp"

#define LMP_AUTOGRAD_FN_BINARY_DECL(args) \
    LMP_AUTOGRAD_FN_BINARY_DECL_HELPER args

#define LMP_AUTOGRAD_FN_BINARY_DECL_HELPER(grad_fn, self_grad, other_grad) \
variable_list grad_fn::apply(const variable_list& gradOutputs) { \
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 2) << "Output size mismatch."; \
  const Variable& grad = gradOutputs[0];                                   \
  Variable& self = (*saved_inputs)[0];                                     \
  Variable& other = (*saved_inputs)[1];                                    \
                                                                           \
  self.incr_grad(self_grad);                                               \
  other.incr_grad(other_grad);                                             \
  return {};                                                               \
}


#define LMP_AUTOGRAD_FFN_BINARY_DECL(args) \
    LMP_AUTOGRAD_FFN_BINARY_DECL_HELPER args

#define LMP_AUTOGRAD_FFN_BINARY_DECL_HELPER(grad_fn, ten_fn, ...) \
tensor::Tensor grad_fn::execute(const variable_list& inputs) const {            \
    LMP_INTERNAL_ASSERT(inputs.size() == 1) << "Function must take one input";  \
    const Variable& self = inputs[0];                                           \
    const Variable& other = inputs[1];                                          \
                                                                                \
    return ten_fn(self.data(), other.data() __VA_OPT__(, __VA_ARGS__));         \
}