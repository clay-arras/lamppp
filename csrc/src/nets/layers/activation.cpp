#include "lamp3/nets/layers/activation.hpp"
#include <limits>
#include "lamp3/autograd/functions/overloads.hpp"
#include "lamp3/autograd/functions/reduct_ops.hpp"
#include "lamp3/autograd/functions/unary_ops.hpp"
#include "lamp3/autograd/variable.hpp"
#include "lamp3/common/assert.hpp"
#include "lamp3/tensor/data_type.hpp"

namespace lmp::nets {

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
autograd::Variable ReLUImpl::forward(const autograd::Variable& x) const {
  return autograd::ops::clamp(x, 0, std::numeric_limits<tensor::Scalar>::max());
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
autograd::Variable SigmoidImpl::forward(const autograd::Variable& x) const {
  return 1 / (1 + autograd::ops::exp(-x));
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
autograd::Variable TanhImpl::forward(const autograd::Variable& x) const {
  autograd::Variable exp_x = autograd::ops::exp(x);
  autograd::Variable nexp_x = autograd::ops::exp(-x);
  return (exp_x - nexp_x) / (exp_x + nexp_x);
}

SoftmaxImpl::SoftmaxImpl(ssize_t dim) : dim_(dim) {};

autograd::Variable SoftmaxImpl::forward(const autograd::Variable& x) const {
  size_t dim_idx = (dim_ < 0)
                       ? x.data().shape().size() + static_cast<size_t>(dim_)
                       : static_cast<size_t>(dim_);
  LMP_CHECK(dim_idx >= 0 && dim_idx < x.data().shape().size())
      << "Dim not in bounds";

  autograd::Variable max_vals = autograd::ops::max(x, dim_idx);
  autograd::Variable x_shifted = x - max_vals;

  autograd::Variable exp_x = autograd::ops::exp(x_shifted);
  return exp_x / autograd::ops::sum(exp_x, dim_idx);
}

}  // namespace lmp::nets