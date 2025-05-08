#pragma once

#include "autograd/engine/native/copy.cuh"
#include "autograd/engine/native/resize.cuh"
#ifndef STORAGE_H
#define STORAGE_H

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
  explicit Storage(void* data, size_t size, DeviceType from_device,
                   DeviceType to_device)
      : impl(std::make_shared<StorageImpl>(data, size, from_device,
                                           to_device)) {}
  explicit Storage(size_t size, DeviceType device)
      : impl(std::make_shared<StorageImpl>(size, device)) {}

  void* data() const;
  size_t size() const;
  DeviceType device() const;

  void resize_(size_t nsize);
  friend std::ostream& operator<<(std::ostream& os, const Storage& obj);

 private:
  class StorageImpl;
  std::shared_ptr<StorageImpl> impl;
};

class Storage::StorageImpl {
 public:
  explicit StorageImpl(void* src, size_t size, DeviceType from_device,
                       DeviceType to_device)
      : size_(size), device_(to_device) {
    data_ptr_ = empty_stub(to_device, size);
    copy_stub(from_device, src, data_ptr_.data, size, to_device);
  }
  explicit StorageImpl(size_t size, DeviceType device)
      : data_ptr_(empty_stub(device, size)), size_(size), device_(device) {}

  void* data() const { return data_ptr_.data; }
  size_t size() const { return size_; }
  DeviceType device() const { return device_; }

  void resize_(size_t nsize) {
    resize_stub(device_, data_ptr_, size_, nsize);
    size_ = nsize;
  }

 private:
  DataPtr data_ptr_;
  size_t size_;
  DeviceType device_;
};

}  // namespace autograd

#endif  // STORAGE_H