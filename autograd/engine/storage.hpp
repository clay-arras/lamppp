#pragma once

#ifndef STORAGE_H
#define STORAGE_H

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include <vector>
#include "autograd/engine/data_type.hpp"
#include <algorithm>
#include <numeric>
#include "dispatch.hpp"

namespace autograd {

class Storage {
public:
  template <typename T>
  Storage(const std::vector<T>& data, const std::vector<size_t>& shape,
                  DataType type)
      : impl(std::make_unique<StorageImpl>(data, shape, type)) {}
  Storage(const std::vector<size_t>& shape, DataType type)
      : impl(std::make_unique<StorageImpl>(shape, type)) {}

  ~Storage();
  Storage(const Storage& other);
  Storage& operator=(const Storage& other);
  Storage(Storage&& other) noexcept;
  Storage& operator=(Storage&& other) noexcept;

  void* data() const;
  size_t size() const;
  DataType type() const;
  const std::vector<size_t>& shape() const;

  friend std::ostream& operator<<(std::ostream& os, const Storage& obj);

private:
  class StorageImpl;
  std::unique_ptr<StorageImpl> impl;
};

class Storage::StorageImpl {
public:
  template <typename T>
  explicit StorageImpl(const std::vector<T>& data,
                       const std::vector<size_t>& shape, DataType type)
      : shape_(shape), type_(type) {
    size_ = data.size();
    DISPATCH_ALL_TYPES(type, [&] {
      data_ptr_ = static_cast<void*>(new scalar_t[size_]);
      std::transform(data.begin(), data.end(), static_cast<scalar_t*>(data_ptr_),
                     [](const T& val) { return static_cast<scalar_t>(val); });
    });
  }
  explicit StorageImpl(const std::vector<size_t>& shape, DataType type)
      : shape_(shape), type_(type) {
    size_ = std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<size_t>());
    DISPATCH_ALL_TYPES(
        type, [&] { data_ptr_ = static_cast<void*>(new scalar_t[size_]); });
  }

  ~StorageImpl() {
    DISPATCH_ALL_TYPES(type_,
                       [&] { delete[] static_cast<scalar_t*>(data_ptr_); });
  }
  StorageImpl(const StorageImpl& other)
      : shape_(other.shape_), type_(other.type_), size_(other.size_) {
    DISPATCH_ALL_TYPES(type_, [&] {
      data_ptr_ = static_cast<void*>(new scalar_t[size_]);
      std::memcpy(data_ptr_, other.data_ptr_, size_ * sizeof(scalar_t));
    });
  }
  StorageImpl& operator=(const StorageImpl& other) {
    if (this != &other) {
      DISPATCH_ALL_TYPES(type_, [&] {
        delete[] static_cast<scalar_t*>(data_ptr_); 
        data_ptr_ = static_cast<void*>(new scalar_t[other.size_]);
        std::memcpy(data_ptr_, other.data_ptr_, size_ * sizeof(scalar_t));
        size_ = other.size_;
        type_ = other.type_;
        shape_ = other.shape_;
      });
    }
    return *this;
  }

  void* data() const { return data_ptr_; }
  size_t size() const { return size_; }
  DataType type() const { return type_; }
  const std::vector<size_t>& shape() const { return shape_; }

private:
  void* data_ptr_;
  size_t size_;
  DataType type_;
  std::vector<size_t> shape_;
};

}  // namespace autograd

#endif  // STORAGE_H