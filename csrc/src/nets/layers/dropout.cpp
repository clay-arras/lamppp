#include "lamp3/autograd/utils/constructor.hpp"
#include "lamp3/autograd/variable.hpp"
#include "lamp3/autograd/functions/overloads.hpp"
#include "lamp3/nets/layers/dropout.hpp"

namespace lmp::nets {

autograd::Variable DropoutImpl::forward(const autograd::Variable& x) const {
    autograd::Variable mask = autograd::rand(x.data().shape());
    return (mask < p_) * x;
}

}