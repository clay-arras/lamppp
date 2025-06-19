#include "lamppp/autograd/constructor.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/autograd/functions/overloads.hpp"
#include "lamppp/tensor/scalar.hpp"
#include "lamppp/nets/layers/dropout.hpp"
#include <limits>

namespace lmp::nets {

autograd::Variable DropoutImpl::forward(const autograd::Variable& x) const {
    autograd::Variable mask = autograd::rand(x.data().shape());
    return (mask < p_) * x;
}

}