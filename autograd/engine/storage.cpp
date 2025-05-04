#include "autograd/engine/storage.hpp"

namespace autograd {

Storage::~Storage() = default;

Storage::Storage(const Storage& other)
    : impl(std::make_unique<StorageImpl>(*other.impl)) {}

Storage& Storage::operator=(const Storage& other) {
  if (this != &other) {
    *impl = *other.impl;
  }
  return *this;
}

Storage::Storage(Storage&& other) noexcept = default;

Storage& Storage::operator=(Storage&& other) noexcept = default;

void* Storage::data() const { return impl->data(); }

size_t Storage::size() const { return impl->size(); }

DataType Storage::type() const { return impl->type(); }

const std::vector<size_t>& Storage::shape() const { return impl->shape(); }

const static size_t kMaxPrintNumel = 20;
std::ostream& operator<<(std::ostream& os, const Storage& obj) {
  os << "Storage(";
  DISPATCH_ALL_TYPES(obj.type(), [&] {
    os << "data=[";
    for (size_t i = 0; i < obj.size(); i++) {
      os << static_cast<const scalar_t*>(obj.data())[i];
      if (i >= kMaxPrintNumel) {
        os << "...";
        break;
      }
      if (i < obj.size() - 1) {
        os << ", ";
      }
    }
    os << "], shape=[";
    for (size_t i = 0; i < obj.shape().size(); i++) {
      os << obj.shape()[i];
      if (i < obj.shape().size() - 1) {
        os << ", ";
      }
    }
    os << "], type=" << typeid(scalar_t).name();
  });
  os << ")";
  return os;
}

}  // namespace autograd