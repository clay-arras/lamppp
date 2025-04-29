#pragma once

#ifndef _TENSOR_H_
#define _TENSOR_H

#include "autograd/engine/backend/cuda_backend.h"
#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <span>
#include <vector>
#include "tensor_impl.h"

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

  template<typename DataType>
  Tensor(const std::vector<DataType>& data, const std::vector<int>& shape)
      : impl_(std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(data, shape)) {}

  template<typename DataType, typename Backend>
  Tensor(const std::vector<DataType>& data, const std::vector<int>& shape)
      : impl_(std::make_shared<TensorImplModel<DataType, Backend>>(data, shape)) {}

  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
  Tensor(const Tensor& other) 
         : impl_(other.impl_->clone()) {}

  std::shared_ptr<TensorImpl>
      impl_;  // TODO: this should probably be a unique ptr

  template <typename DataType, typename Backend>
  static Tensor create(const std::vector<DataType>& data,
                       const std::vector<int>& shape) {
    std::shared_ptr<TensorImpl> impl =
        std::make_shared<TensorImplModel<DataType, Backend>>(data, shape);
    return Tensor(impl);
  }

  const int size() const { return impl_->data_size(); }
  template <typename T>
  std::span<const T> data() const {
    return std::span<const T>(static_cast<const T*>(impl_->data_ptr()), impl_->data_size());
  }
  const std::vector<int>& shape() const { return impl_->shape(); }

  template<typename T>
  void fill(T item) {
    std::cout << "1: " << item << std::endl;
    // Create the item on the stack to avoid memory leaks
    T stack_item = item;
    // Pass the address of the stack item, which will be properly cast by the implementation
    impl_->fill(&stack_item);
  }

  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  Tensor operator>(const Tensor& other) const;
  Tensor operator<(const Tensor& other) const;
  Tensor operator==(const Tensor& other) const;
  Tensor operator!=(const Tensor& other) const;
  Tensor operator>=(const Tensor& other) const;
  Tensor operator<=(const Tensor& other) const;

  Tensor matmul(const Tensor& other) const;
  Tensor transpose() const;

  Tensor log() const;
  Tensor exp() const;
  Tensor relu() const;

  Tensor sum(int axis) const;
  Tensor mean(int axis) const;
  Tensor max(int axis) const;
  Tensor min(int axis) const;

  friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
};

}  // namespace autograd

#endif  // TENSOR_H