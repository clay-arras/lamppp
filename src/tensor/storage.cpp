#include "lamppp/tensor/storage.hpp"
#include "lamppp/tensor/native/memory_ops.hpp"

namespace lmp::tensor {

void* Storage::data() const noexcept {
  return impl_->data();
}

size_t Storage::byte_size() const noexcept {
  return impl_->byte_size();
}

DeviceType Storage::device() const noexcept {
  return impl_->device();
}

void Storage::resize(size_t nsize) {
  impl_->resize(nsize);
}

std::ostream& operator<<(std::ostream& os, const Storage& obj) {
  obj.impl_->print(os);
  return os;
}

void* Storage::StorageImpl::data() const noexcept {
  return data_ptr_.data();
}
size_t Storage::StorageImpl::byte_size() const noexcept {
  return byte_size_;
}
DeviceType Storage::StorageImpl::device() const noexcept {
  return device_;
}

void Storage::StorageImpl::resize(size_t nsize) {
  ops::resize_stub()(device_, data_ptr_, byte_size_, nsize);
  byte_size_ = nsize;
}

void Storage::StorageImpl::print(std::ostream& os) const {
  os << "Storage(dataPtr=" << data() << ", byteSize=" << byte_size()
     << "device=" << device() << ")";
}

}  // namespace lmp::tensor