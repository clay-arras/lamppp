#include "tensor.h"
#include <cassert>
#include <iostream>
#include "backend/eigen_backend.h"
#include "tensor_impl.h"

namespace autograd {

Tensor Tensor::operator+(const Tensor& other) const {
  return Tensor(std::make_shared<TensorImpl>(EigenBackend().add(*impl_, *other.impl_)));
}

Tensor Tensor::operator-(const Tensor& other) const {
  return Tensor(std::make_shared<TensorImpl>(EigenBackend().sub(*impl_, *other.impl_)));
}

Tensor Tensor::operator*(const Tensor& other) const {
  return Tensor(std::make_shared<TensorImpl>(EigenBackend().mul(*impl_, *other.impl_)));
}

Tensor Tensor::operator/(const Tensor& other) const {
  return Tensor(std::make_shared<TensorImpl>(EigenBackend().div(*impl_, *other.impl_)));
}

Tensor Tensor::operator==(const Tensor& other) const {
  return Tensor(); // Placeholder
}

Tensor Tensor::operator!=(const Tensor& other) const {
  return Tensor(); // Placeholder
}

Tensor Tensor::operator>=(const Tensor& other) const {
  return Tensor(); // Placeholder
}

Tensor Tensor::operator<=(const Tensor& other) const {
  return Tensor(); // Placeholder
}

Tensor Tensor::operator>(const Tensor& other) const {
  return Tensor(); // Placeholder
}

Tensor Tensor::operator<(const Tensor& other) const {
  return Tensor(); // Placeholder
}

Tensor Tensor::log() const {
  return Tensor(std::make_shared<TensorImpl>(EigenBackend().log(*impl_)));
}

Tensor Tensor::exp() const {
  return Tensor(std::make_shared<TensorImpl>(EigenBackend().exp(*impl_)));
}

Tensor Tensor::relu() const {
  return Tensor(std::make_shared<TensorImpl>(EigenBackend().relu(*impl_)));
}

Tensor Tensor::matmul(const Tensor& other) const {
  return Tensor(std::make_shared<TensorImpl>(EigenBackend().matmul(*impl_, *other.impl_)));
}

Tensor Tensor::transpose() const {
  return Tensor(std::make_shared<TensorImpl>(EigenBackend().transpose(*impl_)));
}

Tensor Tensor::sum(int axis) const {
  return Tensor(std::make_shared<TensorImpl>(EigenBackend().sum(*impl_, axis)));
}

Tensor Tensor::max(int axis) const {
  return Tensor(); // Placeholder
}

const int kMaxPrintNumel = 20;

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
  os << "Tensor(data=[";
  for (size_t i = 0; i < obj.size(); i++) {
    os << obj.data()[i];
    if (i >= kMaxPrintNumel) {
      os << "...";
      break;
    }
    if (i < obj.size() - 1) {
      os << ", ";
    }
  }
  os << "], shape=[";
  for (size_t i = 0; i < obj.shape().size(); i++) {
    os << obj.shape()[i];
    if (i < obj.shape().size() - 1) {
      os << ", ";
    }
  }
  os << "])";
  return os;
}

}  // namespace autograd