#pragma once

#include <functional>
#include <memory>

namespace lmp::tensor::detail {

struct DataPtr {
  void* data;
  std::shared_ptr<std::function<void(void*)>> deallocator;

  DataPtr() = default;
  DataPtr(void* data_ptr, std::shared_ptr<std::function<void(void*)>> dealloc)
      : data(data_ptr), deallocator(dealloc) {}
  ~DataPtr() {
    if (deallocator) {
      (*deallocator)(data);
    }
  }
  DataPtr(const DataPtr& other)
      : data(other.data), deallocator(other.deallocator) {}
  DataPtr& operator=(const DataPtr& other) {
    DataPtr tmp(other);
    std::swap(data, tmp.data);
    std::swap(deallocator, tmp.deallocator);
    return *this;
  }
};

}  // namespace lmp::tensor::detail