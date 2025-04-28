#include "tensor.h"
#include <cassert>
#include <iostream>

namespace autograd {

Tensor Tensor::operator+(const Tensor& other) const {
  return Tensor(impl_->add(*impl_, *other.impl_));
}

Tensor Tensor::operator-(const Tensor& other) const {
  return Tensor(impl_->sub(*impl_, *other.impl_));
}

Tensor Tensor::operator*(const Tensor& other) const {
  return Tensor(impl_->mul(*impl_, *other.impl_));
}

Tensor Tensor::operator/(const Tensor& other) const {
  return Tensor(impl_->div(*impl_, *other.impl_));
}

Tensor Tensor::operator==(const Tensor& other) const {
  return Tensor(impl_->equal(*impl_, *other.impl_));
}

Tensor Tensor::operator!=(const Tensor& other) const {
  return Tensor(impl_->not_equal(*impl_, *other.impl_));
}

Tensor Tensor::operator>=(const Tensor& other) const {
  return Tensor(impl_->greater_equal(*impl_, *other.impl_));
}

Tensor Tensor::operator<=(const Tensor& other) const {
  return Tensor(impl_->less_equal(*impl_, *other.impl_));
}

Tensor Tensor::operator>(const Tensor& other) const {
  return Tensor(impl_->greater_than(*impl_, *other.impl_));
}

Tensor Tensor::operator<(const Tensor& other) const {
  return Tensor(impl_->less_than(*impl_, *other.impl_));
}

Tensor Tensor::log() const {
  return Tensor(impl_->log(*impl_));
}

Tensor Tensor::exp() const {
  return Tensor(impl_->exp(*impl_));
}

Tensor Tensor::relu() const {
  return Tensor(impl_->relu(*impl_));
}

Tensor Tensor::matmul(const Tensor& other) const {
  return Tensor(impl_->matmul(*impl_, *other.impl_));
}

Tensor Tensor::transpose() const {
  return Tensor(impl_->transpose(*impl_));
}

Tensor Tensor::sum(int axis) const {
  return Tensor(impl_->sum(*impl_, axis));
}

Tensor Tensor::max(int axis) const {
  return Tensor(impl_->max(*impl_, axis));
}

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
  os << "Tensor(impl=" << *obj.impl_;
  os << ")";
  return os;
}

}  // namespace autograd