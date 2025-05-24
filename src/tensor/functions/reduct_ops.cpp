#include "lamppp/tensor/functions/reduct_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(sum_fn, sum_stub);
LMP_DEFINE_DISPATCH(max_fn, max_stub);
LMP_DEFINE_DISPATCH(min_fn, min_stub);

Tensor sum(const Tensor& a, size_t axis) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sum_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), axis)));
}
Tensor max(const Tensor& a, size_t axis) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      max_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), axis)));
}
Tensor min(const Tensor& a, size_t axis) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      min_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), axis)));
}

}  // namespace lmp::tensor::ops
