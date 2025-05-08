#include "tensor_impl.hpp"
#include <cassert>
#include "native/copy.cuh"
#include "native/fill.cuh"

namespace autograd {

void TensorImpl::copy_(TensorImpl other) {
  copy_stub(other.device(), other.data(), data(), other.size(), device());
}

void TensorImpl::fill_(Scalar item) {
  fill_stub(device(), data(), size(), item, type());
}

void TensorImpl::to_(
    DeviceType
        device) {  // NOTE: everything should be destroyed by default destructors
  assert(device != data_.device());
  data_ = Storage(data_.data(), size(), data_.device(), device);
}

TensorImpl TensorImpl::add(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->add(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::sub(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->sub(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::mul(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->mul(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::div(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->div(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::log(const TensorImpl& a) {
  return TensorImpl(a.backend_->log(a.data_), a.backend_);
}

TensorImpl TensorImpl::exp(const TensorImpl& a) {
  return TensorImpl(a.backend_->exp(a.data_), a.backend_);
}

TensorImpl TensorImpl::relu(const TensorImpl& a) {
  return TensorImpl(a.backend_->relu(a.data_), a.backend_);
}

TensorImpl TensorImpl::matmul(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->matmul(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::transpose(const TensorImpl& a) {
  return TensorImpl(a.backend_->transpose(a.data_), a.backend_);
}

TensorImpl TensorImpl::equal(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->equal(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::not_equal(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->not_equal(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::greater_equal(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->greater_equal(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::less_equal(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->less_equal(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::greater_than(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->greater(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::less_than(const TensorImpl& a, const TensorImpl& b) {
  assert((std::is_same_v<std::decay_t<decltype(*a.backend_)>,
                         std::decay_t<decltype(*b.backend_)>>));
  return TensorImpl(a.backend_->less(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::sum(const TensorImpl& a, size_t axis) {
  return TensorImpl(a.backend_->sum(a.data_, axis), a.backend_);
}

TensorImpl TensorImpl::max(const TensorImpl& a, size_t axis) {
  return TensorImpl(a.backend_->max(a.data_, axis), a.backend_);
}

std::ostream& operator<<(std::ostream& os, const TensorImpl& obj) {
  os << "TensorImpl(data_=" << obj.data_;
  os << ", backend=" << obj.backend_;
  os << ")";
  return os;
}

}  // namespace autograd