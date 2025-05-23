#include "lamppp/tensor/tensor.hpp"

#include <iostream>

namespace lmp::tensor {

void* Tensor::data() const noexcept {
  return impl_->data();
}
DataType Tensor::type() const noexcept {
  return impl_->type();
}
DeviceType Tensor::device() const noexcept {
  return impl_->device();
}
const std::vector<size_t>& Tensor::shape() const noexcept {
  return impl_->shape();
}
const std::vector<detail::stride_t>& Tensor::strides() const noexcept {
  return impl_->strides();
}
size_t Tensor::numel() const noexcept {
  return impl_->numel();
}

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
  return Tensor(
      std::make_shared<TensorImpl>(impl_->reshape_(std::move(new_shape))));
}

Tensor Tensor::squeeze(size_t dim) const {
  return Tensor(std::make_shared<TensorImpl>(impl_->squeeze_(dim)));
}

Tensor Tensor::expand_dims(size_t dim) const {
  return Tensor(std::make_shared<TensorImpl>(impl_->expand_dims_(dim)));
}

Scalar Tensor::index(const std::vector<size_t>& idx) {
  return impl_->index_(idx);
}

void Tensor::copy(const Tensor& other) {
  impl_->copy_(*other.impl_);
}

void Tensor::fill(Scalar item) {
  impl_->fill_(item);
}

void Tensor::to(DeviceType device) {
  impl_->to_(device);
}

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
  obj.impl_->print_(os);
  return os;
}

}  // namespace lmp::tensor