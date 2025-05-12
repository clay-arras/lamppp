#include "include/lamppp/tensor/functions/reduct_ops.hpp"
#include "include/lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(sum_stub);
LMP_DEFINE_DISPATCH(mean_stub);
LMP_DEFINE_DISPATCH(max_stub);
LMP_DEFINE_DISPATCH(min_stub);

TensorImpl sum_cpu(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

TensorImpl mean_cpu(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

TensorImpl max_cpu(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

TensorImpl min_cpu(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

TensorImpl sum_cuda(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

TensorImpl mean_cuda(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

TensorImpl max_cuda(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

TensorImpl min_cuda(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

Tensor sum(const Tensor& a, size_t axis) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sum_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), axis)));
}

Tensor mean(const Tensor& a, size_t axis) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      mean_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), axis)));
}

Tensor max(const Tensor& a, size_t axis) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      max_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), axis)));
}

Tensor min(const Tensor& a, size_t axis) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      min_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), axis)));
}

}  // namespace lmp::tensor::ops
