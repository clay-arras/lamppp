#pragma once

#include <functional>
#include <iostream>
#include <memory>

namespace lmp::tensor::detail {

// struct DataPtr {
//   void* data;
//   std::shared_ptr<std::function<void(void*)>> deallocator;

//   DataPtr() = default;
//   DataPtr(void* data_ptr, std::shared_ptr<std::function<void(void*)>> dealloc)
//       : data(data_ptr), deallocator(dealloc) {}
//   ~DataPtr() {
//     if (deallocator) {
//       (*deallocator)(data);
//     }
//   }
//   DataPtr(const DataPtr& other)
//       : data(other.data), deallocator(other.deallocator) {}
//   DataPtr& operator=(const DataPtr& other) {
//     DataPtr tmp(other);
//     std::swap(data, tmp.data);
//     std::swap(deallocator, tmp.deallocator);
//     return *this;
//   }
// };

// TODO: have it crash in the DEBUG mode, otherwise catch it in production
struct DataPtr {
  std::shared_ptr<void> ptr;

  DataPtr() = default;
  DataPtr(void* data_ptr, std::function<void(void*)> dealloc)
      : ptr(data_ptr, [dealloc](void* data) {
          try {
            dealloc(data);
          } catch (const std::exception& e) {
            std::cerr << "DataPtr deleter error: " << e.what() << "\n";
          } catch (...) {
            std::cerr << "DataPtr deleter threw non-std::exception\n";
          }
        }) {}

  void* data() { return ptr.get(); }
  void* data() const { return ptr.get(); }
};

}  // namespace lmp::tensor::detail