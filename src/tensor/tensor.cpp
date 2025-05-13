#include "include/lamppp/tensor/tensor.hpp"
#include <cassert>
#include <iostream>

namespace lmp::tensor {

Tensor Tensor::reshape(std::vector<size_t> new_shape) {
  return Tensor(
      std::make_shared<TensorImpl>(impl_->reshape_(std::move(new_shape))));
}

Tensor Tensor::squeeze(size_t dim) {
  return Tensor(std::make_shared<TensorImpl>(impl_->squeeze_(dim)));
}

Tensor Tensor::expand_dims(size_t dim) {
  return Tensor(std::make_shared<TensorImpl>(impl_->expand_dims_(dim)));
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