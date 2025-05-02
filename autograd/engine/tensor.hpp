#pragma once

#ifndef _TENSOR_H_
#define _TENSOR_H

#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <span>
#include <vector>
#include "autograd/engine/backend/cuda_backend.hpp"
#include "tensor_impl.hpp"

namespace autograd {

class Tensor {
 public:
  Tensor() = default;
  Tensor(const Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}
  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      impl_ = other.impl_->clone();
    }
    return *this;
  }
  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      impl_ = std::move(other.impl_);
    }
    return *this;
  }

  template <typename DataType>
  Tensor(const std::vector<DataType>& data, const std::vector<size_t>& shape)
      : impl_(
            std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
                data, shape)) {}
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
  Tensor(const Tensor& other) : impl_(other.impl_->clone()) {}

  std::shared_ptr<TensorImpl> impl_;  // TODO: this should probably be a unique ptr

  template <typename DataType, typename Backend>
  static Tensor create(const std::vector<DataType>& data,
                       const std::vector<size_t>& shape) {
    std::shared_ptr<TensorImpl> impl =
        std::make_shared<TensorImplModel<DataType, Backend>>(data, shape);
    return Tensor(impl);
  }

  const size_t size() const { return impl_->data_size(); }
  template <typename T>
  std::span<const T> data() const {
    return std::span<const T>(static_cast<const T*>(impl_->data_ptr()),
                              impl_->data_size());
  }
  const std::vector<size_t>& shape() const { return impl_->shape(); }

  template <typename T>
  void fill(T item) {
    any_type gen = item;
    impl_->fill(gen);
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
};

}  // namespace autograd

#endif  // TENSOR_H