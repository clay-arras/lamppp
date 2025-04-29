#include "tensor.h"
#include <cassert>
#include <iostream>

namespace autograd {

Tensor Tensor::operator+(const Tensor& other) const {
  return Tensor(impl_->add(*other.impl_));
}

Tensor Tensor::operator-(const Tensor& other) const {
  return Tensor(impl_->sub(*other.impl_));
}

Tensor Tensor::operator*(const Tensor& other) const {
  return Tensor(impl_->mul(*other.impl_));
}

Tensor Tensor::operator/(const Tensor& other) const {
  return Tensor(impl_->div(*other.impl_));
}

Tensor Tensor::operator==(const Tensor& other) const {
  return Tensor(impl_->equal(*other.impl_));
}

Tensor Tensor::operator!=(const Tensor& other) const {
  return Tensor(impl_->not_equal(*other.impl_));
}

Tensor Tensor::operator>=(const Tensor& other) const {
  return Tensor(impl_->greater_equal(*other.impl_));
}

Tensor Tensor::operator<=(const Tensor& other) const {
  return Tensor(impl_->less_equal(*other.impl_));
}

Tensor Tensor::operator>(const Tensor& other) const {
  return Tensor(impl_->greater_than(*other.impl_));
}

Tensor Tensor::operator<(const Tensor& other) const {
  return Tensor(impl_->less_than(*other.impl_));
}

Tensor Tensor::log() const {
  return Tensor(impl_->log());
}

Tensor Tensor::exp() const {
  return Tensor(impl_->exp());
}

Tensor Tensor::relu() const {
  return Tensor(impl_->relu());
}

Tensor Tensor::matmul(const Tensor& other) const {
  return Tensor(impl_->matmul(*other.impl_));
}

Tensor Tensor::transpose() const {
  return Tensor(impl_->transpose());
}

Tensor Tensor::sum(int axis) const {
  return Tensor(impl_->sum(axis));
}

Tensor Tensor::max(int axis) const {
  return Tensor(impl_->max(axis));
}

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
  os << "Tensor(impl=" << *obj.impl_;
  os << ")";
  return os;
}

}  // namespace autograd