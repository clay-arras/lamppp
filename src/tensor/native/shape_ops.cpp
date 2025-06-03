#include "lamppp/tensor/native/memory_ops.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"
#include "lamppp/tensor/native/shape_ops.hpp"

namespace lmp::tensor::ops {

Tensor reshape(const Tensor& a, std::vector<size_t> new_shape) {
  std::shared_ptr<TensorImpl> impl = detail::UnsafeTensorAccessor::getImpl(a);
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(impl->reshape(std::move(new_shape))));
}

Tensor squeeze(const Tensor& a, size_t dim) {
  auto impl = detail::UnsafeTensorAccessor::getImpl(a);
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(impl->squeeze(dim)));
}

Tensor expand_dims(const Tensor& a, size_t dim) {
  auto impl = detail::UnsafeTensorAccessor::getImpl(a);
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(impl->expand_dims(dim)));
}

}  // namespace lmp::tensor::ops
