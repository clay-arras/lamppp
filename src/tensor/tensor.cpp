#include "include/lamppp/tensor/tensor.hpp"
#include <cassert>
#include <iostream>

namespace lmp::tensor {

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