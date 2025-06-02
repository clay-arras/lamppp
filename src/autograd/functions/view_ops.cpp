#include "lamppp/autograd/functions/view_ops.hpp"
#include <cmath>
#include "lamppp/autograd/functions/unary_decl.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/fill_like.hpp"
#include "lamppp/tensor/native/shape_ops.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::autograd::ops {

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FN_UNARY_DECL,
    ((ToBackward, grad.grad().to(self.data().device())),
     (ReshapeBackward, grad.grad().reshape(self.data().shape())),
     (SqueezeBackward, grad.grad().expand_dims(axis)),
     (ExpandDimsBackward, grad.grad().squeeze(axis)), ));

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FFN_UNARY_DECL,
    ((To, tensor::ops::to, device), (Reshape, tensor::ops::reshape, shape),
     (Squeeze, tensor::ops::squeeze, axis),
     (ExpandDims, tensor::ops::expand_dims, axis), ));

// variable_list ToBackward::apply(const variable_list& gradOutputs) {
//   LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch.";
//   const Variable& grad = gradOutputs[0];
//   Variable& self = (*saved_inputs)[0];

//   self.incr_grad(grad.grad().to(self.data().device()));

//   variable_list grad_inputs = {};
//   return grad_inputs;
// }

// tensor::Tensor To::execute(const variable_list& inputs) {
//   LMP_INTERNAL_ASSERT(inputs.size() == 1) << "Function must take one input";
//   const Variable& self = inputs[0];

//   return self.data().to(device);
// }

// variable_list ReshapeBackward::apply(const variable_list& gradOutputs) {
//   LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch.";
//   const Variable& grad = gradOutputs[0];
//   Variable& self = (*saved_inputs)[0];

//   self.incr_grad(grad.grad().reshape(self.data().shape()));

//   variable_list grad_inputs = {};
//   return grad_inputs;
// }

// tensor::Tensor Reshape::execute(const variable_list& inputs) {
//   LMP_INTERNAL_ASSERT(inputs.size() == 1) << "Function must take one input";
//   const Variable& self = inputs[0];

//   return self.data().reshape(shape);
// }

// variable_list SqueezeBackward::apply(const variable_list& gradOutputs) {
//   LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch.";
//   const Variable& grad = gradOutputs[0];
//   Variable& self = (*saved_inputs)[0];

//   self.incr_grad(grad.grad().expand_dims(axis));

//   variable_list grad_inputs = {};
//   return grad_inputs;
// }

// tensor::Tensor Squeeze::execute(const variable_list& inputs) {
//   LMP_INTERNAL_ASSERT(inputs.size() == 1) << "Function must take one input";
//   const Variable& self = inputs[0];

//   return self.data().squeeze(axis);
// }

// variable_list ExpandDimsBackward::apply(const variable_list& gradOutputs) {
//   LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch.";
//   const Variable& grad = gradOutputs[0];
//   Variable& self = (*saved_inputs)[0];

//   self.incr_grad(grad.grad().squeeze(axis));

//   variable_list grad_inputs = {};
//   return grad_inputs;
// }

// tensor::Tensor ExpandDims::execute(const variable_list& inputs) {
//   LMP_INTERNAL_ASSERT(inputs.size() == 1) << "Function must take one input";
//   const Variable& self = inputs[0];

//   return self.data().expand_dims(axis);
// }

}  // namespace lmp::autograd::ops