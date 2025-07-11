#include "lamppp/autograd/utils/constructor.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/autograd/functions/overloads.hpp"
#include "lamppp/nets/layers/dropout.hpp"

namespace lmp::nets {

autograd::Variable DropoutImpl::forward(const autograd::Variable& x) const {
    autograd::Variable mask = autograd::rand(x.data().shape());
    return (mask < p_) * x;
}

}