#pragma once

#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>
#include "autograd/engine/abstract_backend.hpp"
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/scalar.hpp"
#include "autograd/engine/storage.hpp"

namespace autograd {

class TensorImpl {
 public:
  template <typename T>
  explicit TensorImpl(const std::vector<T>& data,
                      const std::vector<size_t>& shape, DeviceType device,
                      DataType dtype)
      : data_(Storage(static_cast<const void*>(data.data()), data.size(),
                      DeviceType::CPU, device)),
        shape_(shape),
        type_(dtype),
        size_(std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<>())) {}
  explicit TensorImpl(const Storage& storage, const std::vector<size_t>& shape,
                      DataType dtype)
      : data_(storage),
        shape_(shape),
        type_(dtype),
        size_(std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<>())) {}

  void* data() const { return data_.data(); }
  DataType type() const { return type_; };
  DeviceType device() const { return data_.device(); }
  const std::vector<size_t>& shape() const { return shape_; }
  size_t size() const { return size_; }

  void copy_(TensorImpl other);
  void fill_(Scalar item);
  void to_(DeviceType device);
  void print_(std::ostream& os);

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
  size_t size_;
  std::vector<size_t> shape_;
};

}  // namespace autograd