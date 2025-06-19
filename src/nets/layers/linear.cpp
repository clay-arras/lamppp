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

Linear::Linear(size_t in_features, size_t out_features, bool bias, tensor::DeviceType device,
            tensor::DataType dtype) {
    impl_ = std::make_shared<LinearImpl>(LinearImpl(in_features, out_features, bias, device, dtype));
};

}