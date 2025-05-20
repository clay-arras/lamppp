#include "lamppp/tensor/functions/basic_ops.hpp"
#include <array>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/cuda/offset_util.cuh"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(add_stub);
LMP_DEFINE_DISPATCH(sub_stub);
LMP_DEFINE_DISPATCH(mul_stub);
LMP_DEFINE_DISPATCH(div_stub);

TensorImpl add_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}
TensorImpl sub_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}
TensorImpl mul_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}
TensorImpl div_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

Tensor add(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      add_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
               *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor sub(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sub_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
               *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor mul(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      mul_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
               *detail::UnsafeTensorAccessor::getImpl(b))));
}
Tensor div(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      div_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
               *detail::UnsafeTensorAccessor::getImpl(b))));
}

}  // namespace lmp::tensor::ops