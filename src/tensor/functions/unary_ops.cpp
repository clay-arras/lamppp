#include "lamppp/tensor/functions/unary_ops.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(log_stub);
LMP_DEFINE_DISPATCH(exp_stub);
LMP_DEFINE_DISPATCH(sqrt_stub);
LMP_DEFINE_DISPATCH(abs_stub);
LMP_DEFINE_DISPATCH(sin_stub);
LMP_DEFINE_DISPATCH(cos_stub);
LMP_DEFINE_DISPATCH(tan_stub);
LMP_DEFINE_DISPATCH(clamp_stub);

TensorImpl log_cpu(const TensorImpl& a) {
  assert(false && "Not Implemented");
}
TensorImpl exp_cpu(const TensorImpl& a) {
  assert(false && "Not Implemented");
}
TensorImpl sqrt_cpu(const TensorImpl& a) {
  assert(false && "CPU sqrt not implemented");
}
TensorImpl abs_cpu(const TensorImpl& a) {
  assert(false && "CPU abs not implemented");
}
TensorImpl sin_cpu(const TensorImpl& a) {
  assert(false && "CPU sin not implemented");
}
TensorImpl cos_cpu(const TensorImpl& a) {
  assert(false && "CPU cos not implemented");
}
TensorImpl tan_cpu(const TensorImpl& a) {
  assert(false && "CPU tan not implemented");
}
TensorImpl clamp_cpu(const TensorImpl& a, Scalar min_val, Scalar max_val) {
  assert(false && "CPU clamp not implemented");
}

Tensor exp(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      exp_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor log(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      log_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor sqrt(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sqrt_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor abs(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      abs_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor sin(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sin_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor cos(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      cos_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor tan(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      tan_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor clamp(const Tensor& a, Scalar min_val, Scalar max_val) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      clamp_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), min_val,
                 max_val)));
}

}  // namespace lmp::tensor::ops