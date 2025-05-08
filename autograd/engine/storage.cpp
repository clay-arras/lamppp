#include "autograd/engine/storage.hpp"

namespace autograd {

void* Storage::data() const {
  return impl->data();
}

size_t Storage::size() const {
  return impl->size();
}

void Storage::resize_(size_t nsize) {
  impl->resize_(nsize);
}

const static size_t kMaxPrintNumel = 20;
std::ostream& operator<<(std::ostream& os, const Storage& obj) {
  os << "Storage(dataPtr=" << obj.data() << ", size=" << obj.size()
     << "device=" << obj.device() << ")";
  return os;
}

}  // namespace autograd