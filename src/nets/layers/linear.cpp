#include "lamppp/nets/module.hpp"
#include "lamppp/autograd/core.hpp"
#include "lamppp/nets/parameter.hpp"
#include "lamppp/nets/layers/linear.hpp"

namespace lmp::nets {

LinearImpl::LinearImpl(size_t in_features, size_t out_features, bool bias,
                       tensor::DeviceType device, tensor::DataType dtype)
    : requires_bias_(bias),
      weights_(autograd::randn(0, 1, {in_features, out_features}, true, device,
                               dtype)),
      bias_(bias ? autograd::randn(0, 1, {out_features}, true, device, dtype) : Parameter()) {
  register_parameter("[Linear] Weights", weights_);
  if (requires_bias_) {
    register_parameter("[Linear] Bias", bias_);
  }
};

autograd::Variable LinearImpl::forward(const autograd::Variable& x) const {
    if (!requires_bias_) {
        return autograd::ops::matmul(x, weights_);
    }
    return autograd::ops::matmul(x, weights_) + static_cast<autograd::Variable>(bias_);
}

}