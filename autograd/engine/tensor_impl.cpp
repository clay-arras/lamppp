#include "tensor_impl.hpp"
#include <cassert>
#include "autograd/engine/abstract_backend.hpp"
#include "autograd/engine/backend.hpp"
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/dispatch_type.hpp"
#include "native/copy.cuh"
#include "native/fill.cuh"

namespace autograd {

// TODO: this needs to be defined more clearly i.e. what happens if other is bigger/smaller,
// maybe default behavior should be to assign other.type, other.device, other.data COMPLETELY to this
void TensorImpl::copy_(const TensorImpl& other) {
  DISPATCH_ALL_TYPES(other.type(), [&] {
    copy_stub(other.device(), device(), other.data(), data(),
              other.size() * sizeof(scalar_t), other.type(), type());
  });
}

void TensorImpl::fill_(Scalar item) {
  fill_stub(device(), data(), size(), item, type());
}

// NOTE: everything should be destroyed by default destructors
void TensorImpl::to_(DeviceType device) {
  assert(false && "Implement later");
  // assert(device != data_.device());
  // data_ = Storage(data_.data(), size(), data_.device(), device);
}

TensorImpl TensorImpl::add(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).add(a, b);
}

TensorImpl TensorImpl::sub(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).sub(a, b);
}

TensorImpl TensorImpl::mul(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).mul(a, b);
}

TensorImpl TensorImpl::div(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).div(a, b);
}

TensorImpl TensorImpl::log(const TensorImpl& a) {
  return backend_stub(a.device()).log(a);
}

TensorImpl TensorImpl::exp(const TensorImpl& a) {
  return backend_stub(a.device()).exp(a);
}

TensorImpl TensorImpl::relu(const TensorImpl& a) {
  return backend_stub(a.device()).relu(a);
}

TensorImpl TensorImpl::matmul(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).matmul(a, b);
}

TensorImpl TensorImpl::transpose(const TensorImpl& a) {
  return backend_stub(a.device()).transpose(a);
}

TensorImpl TensorImpl::equal(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).equal(a, b);
}

TensorImpl TensorImpl::not_equal(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).not_equal(a, b);
}

TensorImpl TensorImpl::greater_equal(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).greater_equal(a, b);
}

TensorImpl TensorImpl::less_equal(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).less_equal(a, b);
}

TensorImpl TensorImpl::greater_than(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).greater(a, b);
}

TensorImpl TensorImpl::less_than(const TensorImpl& a, const TensorImpl& b) {
  assert(a.device() == b.device());
  return backend_stub(a.device()).less(a, b);
}

TensorImpl TensorImpl::sum(const TensorImpl& a, size_t axis) {
  return backend_stub(a.device()).sum(a, axis);
}

TensorImpl TensorImpl::max(const TensorImpl& a, size_t axis) {
  return backend_stub(a.device()).max(a, axis);
}

const size_t kMaxPrintElem = 1e2;
void TensorImpl::print_(std::ostream& os) {
  os << "Tensor(data=[";
  DISPATCH_ALL_TYPES(this->type_, [&] {
    size_t printSize = std::min(kMaxPrintElem, this->size());
    scalar_t* data_ptr = new scalar_t[printSize * sizeof(scalar_t)];
    copy_stub(this->device(), DeviceType::CPU, this->data(),
              static_cast<void*>(data_ptr), printSize * sizeof(scalar_t),
              type(), type());
    for (size_t i = 0; i < printSize; i++) {
      os << data_ptr[i];
      if (i < printSize - 1) {
        os << ", ";
      } else {
        os << ",...";
      }
    }
  });
  os << "], shape=[";
  for (size_t i = 0; i < shape_.size(); i++) {
    os << shape_[i];
    if (i < shape_.size() - 1) {
      os << ", ";
    }
  }
  os << "], dtype=" << this->type_ << "), device=" << this->device() << ")";
}

}  // namespace autograd