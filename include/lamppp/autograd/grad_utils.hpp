#include "lamppp/tensor/tensor.hpp"

namespace lmp::autograd::detail {

tensor::Tensor sum_broadcast_axis(tensor::Tensor grad,
                                  const std::vector<size_t>& orig_shape);

}