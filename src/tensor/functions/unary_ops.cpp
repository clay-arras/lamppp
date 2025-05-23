#include "lamppp/tensor/functions/unary_ops.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(log_fn, log_stub);
LMP_DEFINE_DISPATCH(exp_fn, exp_stub);
LMP_DEFINE_DISPATCH(sqrt_fn, sqrt_stub);
LMP_DEFINE_DISPATCH(abs_fn, abs_stub);
LMP_DEFINE_DISPATCH(sin_fn, sin_stub);
LMP_DEFINE_DISPATCH(cos_fn, cos_stub);
LMP_DEFINE_DISPATCH(tan_fn, tan_stub);
LMP_DEFINE_DISPATCH(clamp_fn, clamp_stub);

Tensor exp(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      exp_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor log(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      log_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor sqrt(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sqrt_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor abs(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      abs_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor sin(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sin_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor cos(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      cos_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor tan(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      tan_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor clamp(const Tensor& a, Scalar min_val, Scalar max_val) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      clamp_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                   min_val, max_val)));
}

}  // namespace lmp::tensor::ops