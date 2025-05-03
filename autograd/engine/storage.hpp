#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#ifndef STORAGE_H
#define STORAGE_H

#include <numeric>
#include <ostream>
#include <vector>
#include <cstdlib>
#include <cstddef>
#include <memory>
#include "autograd/engine/data_type.hpp"
#include "dispatch.hpp"

namespace autograd {

// Forward declaration of implementation class
class StorageImpl;

struct Storage {
  // Using std::unique_ptr for automatic memory management
  std::unique_ptr<StorageImpl> impl;

  explicit Storage(const std::vector<size_t>& shape, DataType type);

  template<typename T>
  explicit Storage(const std::vector<T>& data, const std::vector<size_t>& shape, DataType type);

  // Destructor (implementation will be in the cpp file)
  ~Storage();

  // Copy and move constructors/assignments
  Storage(const Storage& other);
  Storage& operator=(const Storage& other);
  Storage(Storage&& other) noexcept;
  Storage& operator=(Storage&& other) noexcept;

  // Accessor methods
  void* data() const;
  size_t size() const;
  DataType type() const;
  const std::vector<size_t>& shape() const;

  const static size_t kMaxPrintNumel = 20;
  friend std::ostream& operator<<(std::ostream& os, const Storage& obj) {
    os << "Storage(";
    DISPATCH_ALL_TYPES(obj.type(), [&]{
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
};

}

#endif  // STORAGE_H