#include "lamppp/tensor/functions/expand_ops.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(add_fn, add_stub);
LMP_DEFINE_DISPATCH(sub_fn, sub_stub);
LMP_DEFINE_DISPATCH(mul_fn, mul_stub);
LMP_DEFINE_DISPATCH(div_fn, div_stub);
LMP_DEFINE_DISPATCH(eq_fn, eq_stub);
LMP_DEFINE_DISPATCH(ne_fn, ne_stub);
LMP_DEFINE_DISPATCH(ge_fn, ge_stub);
LMP_DEFINE_DISPATCH(le_fn, le_stub);
LMP_DEFINE_DISPATCH(gt_fn, gt_stub);
LMP_DEFINE_DISPATCH(lt_fn, lt_stub);

TensorImpl add_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_INTERNAL_ASSERT(false, "Not Implemented.");
}
TensorImpl sub_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_INTERNAL_ASSERT(false, "Not Implemented.");
}
TensorImpl mul_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_INTERNAL_ASSERT(false, "Not Implemented.");
}
TensorImpl div_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_INTERNAL_ASSERT(false, "Not Implemented.");
}
TensorImpl eq_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_INTERNAL_ASSERT(false, "Not Implemented.");
}
TensorImpl ne_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_INTERNAL_ASSERT(false, "Not Implemented.");
}
TensorImpl ge_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_INTERNAL_ASSERT(false, "Not Implemented.");
}
TensorImpl le_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_INTERNAL_ASSERT(false, "Not Implemented.");
}
TensorImpl gt_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_INTERNAL_ASSERT(false, "Not Implemented.");
}
TensorImpl lt_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_INTERNAL_ASSERT(false, "Not Implemented.");
}

Tensor add(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      add_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                 *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor sub(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sub_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                 *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor mul(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      mul_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                 *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor div(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      div_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                 *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor equal(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      eq_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor not_equal(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      ne_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor greater_equal(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      ge_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor less_equal(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      le_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor greater(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      gt_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor less(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      lt_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                *detail::UnsafeTensorAccessor::getImpl(b))));
}

}  // namespace lmp::tensor::ops