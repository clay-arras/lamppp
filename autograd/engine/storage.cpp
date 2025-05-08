#include "autograd/engine/storage.hpp"
#include "autograd/engine/native/resize.cuh"

namespace autograd {

void* Storage::data() const {
  return impl->data();
}

size_t Storage::byte_size() const {
  return impl->byte_size();
}

DeviceType Storage::device() const {
  return impl->device();
}

void Storage::resize_(size_t nsize) {
  impl->resize_(nsize);
}

std::ostream& operator<<(std::ostream& os, const Storage& obj) {
  obj.impl->print_(os);
  return os;
}

void Storage::StorageImpl::resize_(size_t nsize) {
  resize_stub(device_, data_ptr_, byte_size_, nsize);
  byte_size_ = nsize;
}

void Storage::StorageImpl::print_(std::ostream& os) {
  os << "Storage(dataPtr=" << data() << ", byteSize=" << byte_size()
     << "device=" << device() << ")";
}

}  // namespace autograd