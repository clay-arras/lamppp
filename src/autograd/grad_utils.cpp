#include "lamppp/autograd/grad_utils.hpp"
#include "lamppp/tensor/native/reduct_ops.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::autograd::detail {

tensor::Tensor sum_broadcast_axis(const tensor::Tensor& grad,
                                  const std::vector<size_t>& orig_shape) {
  if (grad.shape() == orig_shape) {  // no broadcasting
    return grad;
  }

  size_t out_dims = grad.shape().size();
  tensor::Tensor aligned_grad = grad;

#pragma unroll
  for (size_t i = LMP_MAX_DIMS; i-- > 0;) {
    if (i >= out_dims)
      continue;

    int offset = out_dims - 1 - i;
    int a_idx = orig_shape.size() - 1 - offset;
    int a_val = (a_idx >= 0 ? orig_shape[a_idx] : -1);

    if (a_val == 1 || a_val == -1) {
      aligned_grad = tensor::ops::sum(aligned_grad, i);
    }

    while (aligned_grad.shape().size() > orig_shape.size()) {
      aligned_grad = tensor::ops::sum(aligned_grad, 0);
      aligned_grad = aligned_grad.squeeze(0);
    }
  }
  return aligned_grad;
}

}  // namespace lmp::autograd::detail