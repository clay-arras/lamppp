#pragma once

#include "lamppp/autograd/functions/reduct_ops.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/assert.hpp"

#define LMP_AUTOGRAD_FN_UNARY_DECL(args) \
    LMP_AUTOGRAD_FN_UNARY_DECL_HELPER args

#define LMP_AUTOGRAD_FN_UNARY_DECL_HELPER(grad_fn, self_grad) \
variable_list grad_fn::apply(const variable_list& gradOutputs) { \
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch."; \
  const Variable& grad = gradOutputs[0];                                   \
  Variable& self = (*saved_inputs)[0];                                     \
                                                                           \
  if (self.requires_grad()) self.incr_grad(self_grad);                     \
  return {};                                                               \
}


#define LMP_AUTOGRAD_FFN_UNARY_DECL(args) \
    LMP_AUTOGRAD_FFN_UNARY_DECL_HELPER args

#define LMP_AUTOGRAD_FFN_UNARY_DECL_HELPER(grad_fn, ten_fn, ...) \
tensor::Tensor grad_fn::execute(const variable_list& inputs) const {            \
    LMP_INTERNAL_ASSERT(inputs.size() == 1) << "Function must take one input";  \
    const Variable& self = inputs[0];                                           \
    return ten_fn(self.data() __VA_OPT__(, __VA_ARGS__));                       \
}