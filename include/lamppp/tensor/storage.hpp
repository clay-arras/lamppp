#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include "data_ptr.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/native/memory_ops.hpp"

namespace lmp::tensor {

class Storage {
 public:
  explicit Storage(size_t byte_size, DeviceType device)
      : impl_(std::make_shared<StorageImpl>(byte_size, device)) {}

  void* data() const noexcept;
  size_t byte_size() const noexcept;
  DeviceType device() const noexcept;

  void resize_(size_t nsize);
  friend std::ostream& operator<<(std::ostream& os, const Storage& obj);

 private:
  class StorageImpl;
  std::shared_ptr<StorageImpl> impl_;
};

class Storage::StorageImpl {
 public:
  explicit StorageImpl(size_t byte_size, DeviceType device)
      : data_ptr_(ops::empty_stub()(device, byte_size)),
        byte_size_(byte_size),
        device_(device) {}

  void* data() const noexcept;
  size_t byte_size() const noexcept;
  DeviceType device() const noexcept;

  void resize_(size_t nsize);
  void print_(std::ostream& os);

 private:
  detail::DataPtr data_ptr_;
  size_t byte_size_;
  DeviceType device_;
};

}  // namespace lmp::tensor