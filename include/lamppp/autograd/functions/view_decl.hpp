#pragma once

#include "lamppp/autograd/functions/view_ops.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/assert.hpp"

#define LMP_AUTOGRAD_FN_VIEW_DECL(args) \
    LMP_AUTOGRAD_FN_VIEW_DECL_HELPER args

#define LMP_AUTOGRAD_FN_VIEW_DECL_HELPER(grad_fn, grad_expr) \
variable_list grad_fn::apply(const variable_list& gradOutputs) { \
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch."; \
  const Variable& grad = gradOutputs[0]; \
  Variable& self = (*saved_inputs)[0]; \
  \
  self.incr_grad(grad_expr); \
  \
  variable_list grad_inputs = {}; \
  return grad_inputs; \
}

#define LMP_AUTOGRAD_FFN_VIEW_DECL(args) \
    LMP_AUTOGRAD_FFN_VIEW_DECL_HELPER args

#define LMP_AUTOGRAD_FFN_VIEW_DECL_HELPER(fn_name, ten_fn, ...) \
tensor::Tensor fn_name::execute(const variable_list& inputs) { \
  LMP_INTERNAL_ASSERT(inputs.size() == 1) << "Function must take one input"; \
  const Variable& self = inputs[0]; \
  \
  return self.data().ten_fn(__VA_ARGS__); \
}
