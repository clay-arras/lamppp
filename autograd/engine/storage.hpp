#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/native/empty.cuh"
#include "data_ptr.hpp"

namespace autograd {

class Storage {
 public:
  explicit Storage(size_t byte_size, DeviceType device)
      : impl(std::make_shared<StorageImpl>(byte_size, device)) {}

  void* data() const;
  size_t byte_size() const;
  DeviceType device() const;

  void resize_(size_t nsize);
  friend std::ostream& operator<<(std::ostream& os, const Storage& obj);

 private:
  class StorageImpl;
  std::shared_ptr<StorageImpl> impl;
};

class Storage::StorageImpl {
 public:
  explicit StorageImpl(size_t size, DeviceType device)
      : data_ptr_(empty_stub(device, size)),
        byte_size_(size),
        device_(device) {}

  void* data() const { return data_ptr_.data; }
  size_t byte_size() const { return byte_size_; }
  DeviceType device() const { return device_; }

  void resize_(size_t nsize);
  void print_(std::ostream& os);

 private:
  DataPtr data_ptr_;
  size_t byte_size_;
  DeviceType device_;
};

}  // namespace autograd
