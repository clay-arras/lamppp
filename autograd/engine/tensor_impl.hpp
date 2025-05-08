#pragma once

#include "autograd/engine/scalar.hpp"
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

namespace autograd {

class TensorImpl {
 public:  // TODO: FIX THIS WITH DeviceType
  TensorImpl(const Storage& storage, std::shared_ptr<AbstractBackend> backend)
      : data_(storage), backend_(backend) {}

  void* data() const { return data_.data(); }
  std::shared_ptr<AbstractBackend> backend() const { return backend_; }
  DataType type() const { return type_; };
  DeviceType device() const { return data_.device(); }
  const std::vector<size_t>& shape() const { return shape_; }
  size_t size() const { return data_.size(); }

  void copy_(TensorImpl other);
  void fill_(Scalar item);
  void to_(DeviceType device);

  friend std::ostream& operator<<(std::ostream& os, const TensorImpl& obj);

 private:
  friend class Tensor;

  static TensorImpl add(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl sub(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl mul(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl div(const TensorImpl& a, const TensorImpl& b);

  static TensorImpl equal(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl not_equal(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl greater_equal(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl less_equal(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl greater_than(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl less_than(const TensorImpl& a, const TensorImpl& b);

  static TensorImpl log(const TensorImpl& a);
  static TensorImpl exp(const TensorImpl& a);
  static TensorImpl relu(const TensorImpl& a);

  static TensorImpl matmul(const TensorImpl& a, const TensorImpl& b);
  static TensorImpl transpose(const TensorImpl& a);

  static TensorImpl sum(const TensorImpl& a, size_t axis);
  static TensorImpl max(const TensorImpl& a, size_t axis);

  DataType type_;
  Storage data_;
  std::shared_ptr<AbstractBackend> backend_;
  std::vector<size_t> shape_;
};

}  // namespace autograd

#endif  // TENSOR_IMPL_H