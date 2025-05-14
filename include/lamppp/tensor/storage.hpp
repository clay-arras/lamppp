#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include "data_ptr.hpp"
#include "include/lamppp/tensor/device_type.hpp"
#include "include/lamppp/tensor/native/empty.cuh"

namespace lmp::tensor {

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
  explicit StorageImpl(size_t byte_size, DeviceType device)
      : data_ptr_(detail::native::empty_stub(device, byte_size)),
        byte_size_(byte_size),
        device_(device) {}

  void* data() const { return data_ptr_.data; }
  size_t byte_size() const { return byte_size_; }
  DeviceType device() const { return device_; }

  void resize_(size_t nsize);
  void print_(std::ostream& os);

 private:
  detail::DataPtr data_ptr_;
  size_t byte_size_;
  DeviceType device_;
};

}  // namespace lmp::tensor