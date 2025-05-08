#pragma once

#include "autograd/engine/allocator.hpp"
#include "autograd/engine/device_type.hpp"
#ifndef BACKEND_H
#define BACKEND_H

#include <cstddef>

namespace autograd {

struct Storage;

class AbstractBackend {
 public:
  virtual Storage add(const Storage& a, const Storage& b) = 0;
  virtual Storage sub(const Storage& a, const Storage& b) = 0;
  virtual Storage mul(const Storage& a, const Storage& b) = 0;
  virtual Storage div(const Storage& a, const Storage& b) = 0;

  virtual Storage log(const Storage& a) = 0;
  virtual Storage exp(const Storage& a) = 0;
  virtual Storage relu(const Storage& a) = 0;

  virtual Storage matmul(const Storage& a, const Storage& b) = 0;
  virtual Storage transpose(const Storage& a) = 0;

  virtual Storage equal(const Storage& a, const Storage& b) = 0;
  virtual Storage not_equal(const Storage& a, const Storage& b) = 0;
  virtual Storage greater_equal(const Storage& a, const Storage& b) = 0;
  virtual Storage less_equal(const Storage& a, const Storage& b) = 0;
  virtual Storage greater(const Storage& a, const Storage& b) = 0;
  virtual Storage less(const Storage& a, const Storage& b) = 0;

  virtual Storage sum(const Storage& a, size_t axis) = 0;
  virtual Storage max(const Storage& a, size_t axis) = 0;
};

template <typename Derived>
class Singleton {
 public:
  static Derived& instance() {
    static Derived instance;
    return instance;
  }

  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;
  Singleton(Singleton&&) = delete;
  Singleton& operator=(Singleton&&) = delete;

 protected:
  Singleton() = default;
  ~Singleton() = default;
};

}  // namespace autograd

#endif  // BACKEND_H