#include "lamppp/autograd/functions/unary_ops.hpp"
#include "lamppp/tensor/scalar.hpp"
#include "lamppp/nets/layers/activation.hpp"
#include <limits>

namespace lmp::nets {

autograd::Variable ReLUImpl::forward(const autograd::Variable& x) const {
    return autograd::ops::clamp(x, 0, std::numeric_limits<tensor::Scalar>::max());
}

}