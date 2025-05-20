#include "lamppp/tensor/functions/binary_ops.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(eq_stub);
LMP_DEFINE_DISPATCH(ne_stub);
LMP_DEFINE_DISPATCH(ge_stub);
LMP_DEFINE_DISPATCH(le_stub);
LMP_DEFINE_DISPATCH(gt_stub);
LMP_DEFINE_DISPATCH(lt_stub);

TensorImpl eq_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}
TensorImpl ne_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}
TensorImpl ge_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}
TensorImpl le_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}
TensorImpl gt_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}
TensorImpl lt_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

Tensor equal(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      eq_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
              *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor not_equal(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      ne_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
              *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor greater_equal(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      ge_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
              *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor less_equal(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      le_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
              *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor greater(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      gt_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
              *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor less(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      lt_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
              *detail::UnsafeTensorAccessor::getImpl(b))));
}

}  // namespace lmp::tensor::ops