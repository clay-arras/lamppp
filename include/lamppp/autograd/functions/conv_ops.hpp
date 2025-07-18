#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"

namespace lmp::autograd::ops {

// /// @internal
// struct ConvolutionBackward : public Function { // TODO(nx2372): need to change this to "cross correlation"
//   size_t stride_, padding_, dilation_;
//   explicit ConvolutionBackward(size_t stride, size_t padding, size_t dilation)
//       : stride_(stride), padding_(padding), dilation_(dilation) {}
//   variable_list apply(const variable_list& gradOutputs) override;
// };

// struct Convolution : public ForwardFunction<Convolution> {
//   using DefaultBackward = ConvolutionBackward;
//   size_t stride_, padding_, dilation_;
//   explicit Convolution(size_t stride, size_t padding, size_t dilation)
//       : stride_(stride), padding_(padding), dilation_(dilation) {}
//   tensor::Tensor execute(const variable_list& inputs);
// };
// /// @endinternal

// inline Variable conv(const Variable& input, const Variable& kernel,
//                      size_t stride, size_t padding, size_t dilation) {
//   return VariableOpFact::apply<Convolution>({input, kernel}, stride, padding,
//                                             dilation)[0];
// }

}  // namespace lmp::autograd::ops