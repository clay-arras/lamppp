#pragma once

#include <functional>
#include <memory>

namespace autograd {

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
  //   DataPtr(const DataPtr& other) {}
  //   DataPtr& operator=(const DataPtr& other) {}
};

}  // namespace autograd