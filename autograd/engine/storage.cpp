#include "autograd/engine/storage.hpp"
#include "autograd/engine/native/resize.cuh"

namespace autograd {

void* Storage::data() const {
  return impl->data();
}

size_t Storage::size() const {
  return impl->size();
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
  resize_stub(device_, data_ptr_, size_, nsize);
  size_ = nsize;
}

void Storage::StorageImpl::print_(std::ostream& os) {
  os << "Storage(dataPtr=" << data() << ", size=" << size()
     << "device=" << device() << ")";
}

}  // namespace autograd