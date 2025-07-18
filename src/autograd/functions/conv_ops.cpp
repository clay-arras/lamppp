#include "lamppp/autograd/functions/conv_ops.hpp"
#include "lamppp/autograd/function.hpp"
#include "lamppp/autograd/variable.hpp"

namespace lmp::autograd::ops {

// variable_list ConvolutionBackward::apply(
//     const variable_list& gradOutputs) {
//   LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch.";
//   const Variable& grad = gradOutputs[0];
//   Variable& self = (*saved_inputs)[0];
//   Variable& other = (*saved_inputs)[1];

// //   if (self.requires_grad())
// //     self.incr_grad();
// //   if (other.requires_grad())
// //     other.incr_grad();

//   variable_list grad_inputs = {};
//   return grad_inputs;
// }

// tensor::Tensor Convolution::execute(const variable_list& inputs) {
//   LMP_INTERNAL_ASSERT(inputs.size() == 2) << "Function must take 2 inputs";
//   const Variable& self = inputs[0];
//   const Variable& other = inputs[1];

//   return tensor::ops::conv(self.data(), other.data(), stride_, padding_, dilation_);
// }

}  // namespace lmp::autograd::ops