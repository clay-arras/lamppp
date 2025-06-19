#include "lamppp/nets/module.hpp"
#include "lamppp/autograd/core.hpp"
#include "lamppp/nets/layers/linear.hpp"

namespace lmp::nets {

autograd::Variable LinearImpl::forward(const autograd::Variable& x) const {
    if (!requires_bias_) {
        return autograd::ops::matmul(x, weights_);
    }
    return autograd::ops::matmul(x, weights_) + static_cast<autograd::Variable>(bias_);
}

}