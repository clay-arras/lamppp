#include "lamppp/tensor/tensor_impl.hpp"
#include <cassert>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_type.hpp"
#include "lamppp/tensor/native/copy.cuh"
#include "lamppp/tensor/native/fill.cuh"

namespace lmp::tensor {

// TODO: can potentially lazy initialize strides, if you never use it for aligneding
TensorImpl::TensorImpl(const Storage& storage, const std::vector<size_t>& shape,
                       DataType dtype)
    : data_(storage),
      shape_(shape),
      type_(dtype),
      strides_(std::vector<detail::stride_t>(shape.size())),
      size_(shape.empty() ? 0
                          : std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<>())) {
  LMP_DISPATCH_ALL_TYPES(dtype, [&] {
    assert(data_.byte_size() / sizeof(scalar_t) == size_ &&
           "Size mismatch, product of shape must equal num elements");
  });
  update_strides_();
}

void* TensorImpl::data() const noexcept {
  return data_.data();
}
DataType TensorImpl::type() const noexcept {
  return type_;
}
DeviceType TensorImpl::device() const noexcept {
  return data_.device();
}
const std::vector<size_t>& TensorImpl::shape() const noexcept {
  return shape_;
}
const std::vector<detail::stride_t>& TensorImpl::strides() const noexcept {
  return strides_;
}
size_t TensorImpl::size() const noexcept {
  return size_;
}

void TensorImpl::update_strides_() {
  detail::stride_t stride = 1;
  for (int i = shape_.size() - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= shape_[i];
  }
}

TensorImpl TensorImpl::reshape_(std::vector<size_t> new_shape) {
  size_t new_size = new_shape.empty()
                        ? 0
                        : std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                          std::multiplies<>());
  assert(
      new_size == size_ &&
      "Cannot reshape tensor: total number of elements must remain the same.");
  TensorImpl other(*this);
  other.shape_ = std::move(new_shape);
  other.update_strides_();
  return other;
}

TensorImpl TensorImpl::squeeze_(size_t dim) {
  assert(dim < shape_.size() && "Dimension out of range for squeeze");
  assert(shape_[dim] == 1 && "Cannot squeeze dimension that is not size 1");
  TensorImpl other(*this);
  other.shape_.erase(other.shape_.begin() + dim);
  other.update_strides_();
  return other;
}

TensorImpl TensorImpl::expand_dims_(size_t dim) {
  assert(dim <= shape_.size() && "Dimension out of range for expand_dims");
  TensorImpl other(*this);
  other.shape_.insert(other.shape_.begin() + dim, 1);
  other.update_strides_();
  return other;
}

// TODO: this needs to be defined more clearly i.e. what happens if other is bigger/smaller,
// maybe default behavior should be to assign other.type, other.device, other.data COMPLETELY to this
void TensorImpl::copy_(const TensorImpl& other) {
  LMP_DISPATCH_ALL_TYPES(other.type(), [&] {
    detail::native::copy_stub(other.device(), device(), other.data(), data(),
                              other.size() * sizeof(scalar_t), other.type(),
                              type());
  });
}

void TensorImpl::fill_(Scalar item) {
  detail::native::fill_stub(device(), data(), size(), item, type());
}

// NOTE: everything should be destroyed by default destructors
void TensorImpl::to_(DeviceType device) {
  assert(false && "Implement later");
  // assert(device != data_.device());
  // data_ = Storage(data_.data(), size(), data_.device(), device);
}

const size_t kMaxPrintElem = 1e2;
void TensorImpl::print_(std::ostream& os) {
  os << "Tensor(data=[";
  LMP_DISPATCH_ALL_TYPES(this->type_, [&] {
    size_t printSize = std::min(kMaxPrintElem, this->size());
    scalar_t* data_ptr = new scalar_t[printSize * sizeof(scalar_t)];
    detail::native::copy_stub(this->device(), DeviceType::CPU, this->data(),
                              static_cast<void*>(data_ptr), printSize,
                              this->type_, this->type_);
    for (size_t i = 0; i < printSize; i++) {
      os << data_ptr[i];
      if (i < printSize - 1) {
        os << ", ";
      } else if (printSize < this->size()) {
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

}  // namespace lmp::tensor