#include "lamppp/autograd/functions/unary_ops.hpp"
#include "lamppp/autograd/functions/overloads.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/tensor/scalar.hpp"
#include "lamppp/nets/layers/activation.hpp"
#include <limits>

namespace lmp::nets {

autograd::Variable ReLUImpl::forward(const autograd::Variable& x) const {
    return autograd::ops::clamp(x, 0, std::numeric_limits<tensor::Scalar>::max());
}

autograd::Variable SigmoidImpl::forward(const autograd::Variable& x) const {
    return 1 / (1 + autograd::ops::exp(-x));
}

autograd::Variable TanhImpl::forward(const autograd::Variable& x) const {
    autograd::Variable exp_x = autograd::ops::exp(x);
    autograd::Variable nexp_x = autograd::ops::exp(-x);
    return (exp_x - nexp_x) / (exp_x + nexp_x);
}

}