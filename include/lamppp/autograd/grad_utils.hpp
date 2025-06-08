#include "lamppp/tensor/tensor.hpp"

namespace lmp::autograd::detail {

/// @internal
/**
 * @brief Sum the gradient along a specific axis
 * @param grad The gradient tensor
 * @param orig_shape The original shape of the tensor
 * @return The summed gradient tensor
 */
tensor::Tensor sum_broadcast_axis(const tensor::Tensor& grad,
                                  const std::vector<size_t>& orig_shape);

/// @endinternal

}  // namespace lmp::autograd::detail