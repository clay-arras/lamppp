#include "lamppp/autograd/functions/view_ops.hpp"
#include "lamppp/common/assert.hpp"
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
    if (!requires_bias_) { // (batch x in) x (in x out)
        return autograd::ops::matmul(x, weights_);
    }
    return autograd::ops::matmul(x, weights_) + static_cast<autograd::Variable>(bias_);
}

FlattenImpl::FlattenImpl(ssize_t start_dim, ssize_t end_dim) : start_dim_(start_dim), end_dim_(end_dim) {
  LMP_CHECK(start_dim_ >= 0);
};

autograd::Variable FlattenImpl::forward(const autograd::Variable& x) const {
  ssize_t end_dim_idx = (end_dim_ < 0) ? x.data().shape().size() + end_dim_ : end_dim_;
  LMP_CHECK(start_dim_ < end_dim_idx)
      << "Start and end dims not valid";

  size_t flattened_dims = 1;
  std::vector<size_t> new_shape;

  for (ssize_t i = 0; i < x.data().shape().size(); i++) {
    if (i < start_dim_ || i > end_dim_idx) {
      new_shape.push_back(x.data().shape()[i]);
      continue;
    }
    flattened_dims *= x.data().shape()[i];
    if (i == end_dim_idx) {
      new_shape.push_back(flattened_dims);
    }
  }

  return autograd::ops::reshape(x, new_shape);
};

}