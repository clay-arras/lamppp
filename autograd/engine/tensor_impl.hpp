#pragma once

#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include "autograd/engine/backend.hpp"
#include <cstddef>
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/storage.hpp"
#include "scalar.hpp"

namespace autograd {

class TensorImpl {
public:
  TensorImpl() = delete;
  ~TensorImpl() = default;

  void* data() const { return data_.data(); }
  DataType type() const { return data_.type(); };
  const std::vector<size_t>& shape() const { return data_.shape(); }
  size_t size() const { return data_.size(); }

  TensorImpl(const Storage& storage, std::shared_ptr<AbstractBackend> backend)
      : data_(storage), backend_(backend) {}

  TensorImpl add(const TensorImpl& a, const TensorImpl& b);
  TensorImpl sub(const TensorImpl& a, const TensorImpl& b);
  TensorImpl mul(const TensorImpl& a, const TensorImpl& b);
  TensorImpl div(const TensorImpl& a, const TensorImpl& b);

  TensorImpl log(const TensorImpl& a);
  TensorImpl exp(const TensorImpl& a);
  TensorImpl relu(const TensorImpl& a);

  TensorImpl matmul(const TensorImpl& a, const TensorImpl& b);
  TensorImpl transpose(const TensorImpl& a);

  TensorImpl equal(const TensorImpl& a, const TensorImpl& b);
  TensorImpl not_equal(const TensorImpl& a, const TensorImpl& b);
  TensorImpl greater_equal(const TensorImpl& a, const TensorImpl& b);
  TensorImpl less_equal(const TensorImpl& a, const TensorImpl& b);
  TensorImpl greater_than(const TensorImpl& a, const TensorImpl& b);
  TensorImpl less_than(const TensorImpl& a, const TensorImpl& b);

  TensorImpl sum(const TensorImpl& a, size_t axis);
  TensorImpl max(const TensorImpl& a, size_t axis);

  inline void fill(Scalar t) {
    DISPATCH_ALL_TYPES(type(), [&]{
      scalar_t* st = static_cast<scalar_t*>(data());
      std::fill(st, st + data_.size(), static_cast<scalar_t>(t));
    });
  }

  friend std::ostream& operator<<(std::ostream& os, const TensorImpl& obj) {
    os << "TensorImpl(data_=" << obj.data_;
    os << ", backend=" << obj.backend_;
    os << ")";
    return os;
  }
    
// private:
  std::shared_ptr<AbstractBackend> backend_; 
  Storage data_;
};

}  // namespace autograd

#endif  // TENSOR_IMPL_H