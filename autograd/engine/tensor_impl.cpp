#include "tensor_impl.hpp"

namespace autograd {

TensorImpl TensorImpl::add(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->add(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::sub(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->sub(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::mul(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->mul(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::div(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->div(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::log(const TensorImpl& a) {
  return TensorImpl(backend_->log(a.data_), a.backend_);
}

TensorImpl TensorImpl::exp(const TensorImpl& a) {
  return TensorImpl(backend_->exp(a.data_), a.backend_);
}

TensorImpl TensorImpl::relu(const TensorImpl& a) {
  return TensorImpl(backend_->relu(a.data_), a.backend_);
}

TensorImpl TensorImpl::matmul(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->matmul(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::transpose(const TensorImpl& a) {
  return TensorImpl(backend_->transpose(a.data_), a.backend_);
}

TensorImpl TensorImpl::equal(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->equal(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::not_equal(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->not_equal(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::greater_equal(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->greater_equal(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::less_equal(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->less_equal(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::greater_than(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->greater_than(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::less_than(const TensorImpl& a, const TensorImpl& b) {
  return TensorImpl(backend_->less_than(a.data_, b.data_), a.backend_);
}

TensorImpl TensorImpl::sum(const TensorImpl& a, size_t axis) {
  return TensorImpl(backend_->sum(a.data_, axis), a.backend_);
}

TensorImpl TensorImpl::max(const TensorImpl& a, size_t axis) {
  return TensorImpl(backend_->max(a.data_, axis), a.backend_);
}

}