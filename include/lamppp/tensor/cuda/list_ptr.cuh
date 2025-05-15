#pragma once

#include <driver_types.h>
#include <cassert>
#include <memory>

namespace lmp::tensor::detail::cuda {

template <typename T>
class ListDevicePtr {
 private:
  std::shared_ptr<T[]> ptr_;
  size_t size_;

 public:
  ListDevicePtr() = default;
  explicit ListDevicePtr(const T* obj_list, size_t size) : size_(size) {
    T* raw = nullptr;
    cudaError_t err = cudaMalloc(&raw, sizeof(T) * size);
    assert(err == cudaSuccess && "ListDevicePtr: cudaMalloc failed");
    err = cudaMemcpy(raw, obj_list, sizeof(T) * size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess && "ListDevicePtr: cudaMemcpy failed");
    ptr_ = std::shared_ptr<T[]>(raw, [](T* p) {
      cudaError_t err = cudaFree(p);
      assert(err == cudaSuccess && "ListDevicePtr: cudaFree failed");
    });
  }

  T* get() const noexcept { return ptr_.get(); }
  size_t size() const noexcept { return size_; }
};

}  // namespace lmp::tensor::detail::cuda