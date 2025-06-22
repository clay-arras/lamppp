#include "lamppp/tensor/tensor_impl.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_type.hpp"
#include "lamppp/tensor/native/memory_ops.hpp"

namespace lmp::tensor {

TensorImpl::TensorImpl(Storage storage, const std::vector<size_t>& shape,
                       DataType dtype)
    : data_(std::move(storage)),
      shape_(shape),
      type_(dtype),
      strides_(std::vector<detail::stride_t>(shape.size())),
      numel_(shape.empty() ? 0
                           : std::accumulate(shape.begin(), shape.end(), 1,
                                             std::multiplies<>())) {
  LMP_DISPATCH_ALL_TYPES(dtype, [&] {
    LMP_CHECK(data_.byte_size() / sizeof(scalar_t) == numel_)
        << "Storage size mismatch: expected " << numel_ << " elements of type "
        << dtype << " (" << sizeof(scalar_t) << " bytes each), but storage has "
        << data_.byte_size() << " bytes (capacity for "
        << (data_.byte_size() / sizeof(scalar_t)) << " elements)";
  });
  update_strides();
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
size_t TensorImpl::numel() const noexcept {
  return numel_;
}

void TensorImpl::update_strides() {
  detail::stride_t stride = 1;
  strides_.resize(shape_.size());
  for (int i = shape_.size() - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= shape_[i];
  }
}

TensorImpl TensorImpl::reshape(std::vector<size_t> new_shape) {
  size_t new_size = new_shape.empty()
                        ? 0
                        : std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                          std::multiplies<>());
  LMP_CHECK(new_size == numel_) << "Cannot reshape tensor: total number of "
                                   "elements must remain the same.";
  TensorImpl other(*this);
  other.shape_ = std::move(new_shape);
  other.update_strides();
  return other;
}

TensorImpl TensorImpl::squeeze(size_t dim) {
  LMP_CHECK(dim < shape_.size()) << "Dimension out of range for squeeze";
  LMP_CHECK(shape_[dim] == 1) << "Cannot squeeze dimension that is not size 1";
  TensorImpl other(*this);
  other.shape_.erase(other.shape_.begin() + dim);
  other.update_strides();
  return other;
}

TensorImpl TensorImpl::expand_dims(size_t dim) {
  LMP_CHECK(dim <= shape_.size()) << "Dimension out of range for expand_dims";
  TensorImpl other(*this);
  other.shape_.insert(other.shape_.begin() + dim, 1);
  other.update_strides();
  return other;
}

Scalar TensorImpl::index(const std::vector<size_t>& idx) {
  LMP_CHECK(idx.size() == shape_.size()) << "Indexing does not match shape";
  size_t at = 0;
  for (size_t i = 0; i < idx.size(); i++) {
    at += strides_[i] * idx[i];
  }
  return LMP_DISPATCH_ALL_TYPES(type(), [&]() {
    auto* elem = new scalar_t[1];
    ops::copy_stub()(device(), DeviceType::CPU,
                     static_cast<scalar_t*>(data()) + at, elem, 1, type(),
                     type());
    return static_cast<Scalar>(elem[0]);
  });
}

// TODO(astronaut): this needs to be defined more clearly i.e. what happens if other is bigger/smaller,
// maybe default behavior should be to assign other.type, other.device, other.data COMPLETELY to this
void TensorImpl::copy(const TensorImpl& other) const {
  LMP_DISPATCH_ALL_TYPES(other.type(), [&] {
    ops::copy_stub()(other.device(), device(), other.data(), data(),
                     other.numel(), other.type(), type());
  });
}

void TensorImpl::fill(Scalar item) const {
  ops::fill_stub()(device(), data(), numel(), item, type());
}

const size_t kMaxPrintElem = 2e1;
void TensorImpl::print(std::ostream& os) const {
  os << "Tensor(data=[";
  LMP_DISPATCH_ALL_TYPES(this->type_, [&] {
    size_t print_size = std::min(kMaxPrintElem, this->numel());
    auto* data_ptr = new scalar_t[print_size * sizeof(scalar_t)];
    ops::copy_stub()(this->device(), DeviceType::CPU, this->data(),
                     static_cast<void*>(data_ptr), print_size, this->type_,
                     this->type_);
    for (size_t i = 0; i < print_size; i++) {
      os << data_ptr[i];
      if (i < print_size - 1) {
        os << ", ";
      } else if (print_size < this->numel()) {
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