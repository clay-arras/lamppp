#pragma once

#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>
#include "autograd/engine/backend.hpp"
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/storage.hpp"
#include "scalar.hpp"
#include "dispatch.hpp"

namespace autograd {

class TensorImpl {
 public:
  TensorImpl(const Storage& storage, std::shared_ptr<AbstractBackend> backend)
      : data_(storage), backend_(backend) {}

  ~TensorImpl() = default;
  TensorImpl(const TensorImpl& other) : data_(other.data_), backend_(other.backend_) {}
  TensorImpl& operator=(const TensorImpl& other) {
    if (this != &other) {
      data_ = other.data_;
      backend_ = other.backend_;
    }
    return *this;
  }
  TensorImpl(TensorImpl&& other) noexcept : data_(std::move(other.data_)), backend_(std::move(other.backend_)) {}
  TensorImpl& operator=(TensorImpl&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      backend_ = std::move(other.backend_);
    }
    return *this;
  }

  void* data() const { return data_.data(); }
  DataType type() const { return data_.type(); };
  const std::vector<size_t>& shape() const { return data_.shape(); }
  size_t size() const { return data_.size(); }
  std::shared_ptr<AbstractBackend> backend() const { return backend_; }

  friend std::ostream& operator<<(std::ostream& os, const TensorImpl& obj);

private:
  friend class Tensor;

  static TensorImpl add(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl sub(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl mul(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl div(const TensorImpl& a, const TensorImpl& b);

  static TensorImpl log(const TensorImpl& a);
  static TensorImpl exp(const TensorImpl& a);
  static TensorImpl relu(const TensorImpl& a);

  static TensorImpl matmul(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl transpose(const TensorImpl& a);

  static TensorImpl equal(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl not_equal(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl greater_equal(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl less_equal(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl greater_than(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl less_than(const TensorImpl& a, const TensorImpl& b);

  static TensorImpl sum(const TensorImpl& a, size_t axis);
  static TensorImpl max(const TensorImpl& a, size_t axis);

  inline void fill(Scalar t) {
    DISPATCH_ALL_TYPES(type(), [&] {
      scalar_t* st = static_cast<scalar_t*>(data());
      std::fill(st, st + data_.size(), static_cast<scalar_t>(t));
    });
  }

  std::shared_ptr<AbstractBackend> backend_;
  Storage data_;
};

}  // namespace autograd

#endif  // TENSOR_IMPL_H