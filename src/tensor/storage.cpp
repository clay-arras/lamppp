#include "lamppp/tensor/storage.hpp"
#include "lamppp/tensor/native/resize.cuh"

namespace lmp::tensor {

void* Storage::data() const noexcept {
  return impl->data();
}

size_t Storage::byte_size() const noexcept {
  return impl->byte_size();
}

DeviceType Storage::device() const noexcept {
  return impl->device();
}

void Storage::resize_(size_t nsize) {
  impl->resize_(nsize);
}

std::ostream& operator<<(std::ostream& os, const Storage& obj) {
  obj.impl->print_(os);
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

void Storage::StorageImpl::resize_(size_t nsize) {
  detail::native::resize_stub()(device_, data_ptr_, byte_size_, nsize);
  byte_size_ = nsize;
}

void Storage::StorageImpl::print_(std::ostream& os) {
  os << "Storage(dataPtr=" << data() << ", byteSize=" << byte_size()
     << "device=" << device() << ")";
}

}  // namespace lmp::tensor