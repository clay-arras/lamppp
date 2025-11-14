#include "lamppp/autograd/utils/grad_utils.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/native/reduct_ops.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/utils/align_utils.hpp"

namespace lmp::autograd::detail {

tensor::Tensor sum_broadcast_axis(const tensor::Tensor& grad,
                                  const std::vector<size_t>& orig_shape) {
  if (grad.shape() == orig_shape) {  // no broadcasting
    return grad;
  }

  size_t out_dims = grad.shape().size();
  tensor::Tensor aligned_grad = grad;

  while (aligned_grad.shape().size() > orig_shape.size()) {
    aligned_grad = tensor::ops::sum(aligned_grad, 0);
    aligned_grad = aligned_grad.squeeze(0);
  }

#pragma unroll
  for (size_t i = 0; i < LMP_MAX_DIMS; i++) {
    if (i >= orig_shape.size())
      continue;

    if (orig_shape[i] == 1) {
      aligned_grad = tensor::ops::sum(aligned_grad, i);
    }
  }
  LMP_INTERNAL_ASSERT(aligned_grad.shape() == orig_shape)
      << "Shape not correct: " << aligned_grad.shape()
      << ". Expected: " << orig_shape;
  return aligned_grad;
}

}  // namespace lmp::autograd::detail