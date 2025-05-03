#pragma once

#ifndef _TENSOR_H_
#define _TENSOR_H

#include <span>
#include "autograd/engine/backend.hpp"
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/storage.hpp"
#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <vector>
#include "tensor_impl.hpp"
#include "dispatch.hpp"

namespace autograd {

class Tensor {
public:
  Tensor() = default;
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
  std::shared_ptr<TensorImpl> impl_;

  void* data() const { return impl_->data(); }
  DataType type() const { return impl_->type(); }
  const std::vector<size_t>& shape() const { return impl_->shape(); }
  size_t size() const { return impl_->size(); }
  std::shared_ptr<AbstractBackend> backend() const { return impl_->backend_; }

  template <typename T>
  static Tensor create(const std::vector<T>& data,
                       const std::vector<size_t>& shape, 
                       std::shared_ptr<AbstractBackend> backend, DataType dtype) {
    std::shared_ptr<TensorImpl> impl =
        std::make_shared<TensorImpl>(Storage(data, shape, dtype), backend);
    return Tensor(impl);
  }

  void fill(Scalar item) {
    impl_->fill(item);
  }

  // const std::vector<size_t>& shape() const { return impl_->shape(); }
  
  template <typename T>
  std::span<T> data() const {
    static thread_local std::vector<T> converted_data;
    DISPATCH_ALL_TYPES(impl_->type(), [&]{
      converted_data.resize(impl_->size());
      scalar_t* original_data = static_cast<scalar_t*>(impl_->data());
      for (size_t i = 0; i < impl_->size(); ++i) {
        converted_data[i] = static_cast<T>(original_data[i]);
      }
    });
    return std::span<T>(converted_data);
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
};

// class Tensor {
//  public:
//   Tensor() = default;
//   Tensor(const Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}
//   Tensor& operator=(const Tensor& other) {
//     if (this != &other) {
//       impl_ = other.impl_->clone();
//     }
//     return *this;
//   }
//   Tensor& operator=(Tensor&& other) noexcept {
//     if (this != &other) {
//       impl_ = std::move(other.impl_);
//     }
//     return *this;
//   }

//   template <typename DataType>
//   Tensor(const std::vector<DataType>& data, const std::vector<size_t>& shape)
//       : impl_(
//             std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
//                 data, shape)) {}
//   explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
//   Tensor(const Tensor& other) : impl_(other.impl_->clone()) {}

//   std::shared_ptr<TensorImpl> impl_;  // TODO: this should probably be a unique ptr

//   template <typename DataType, typename Backend>
//   static Tensor create(const std::vector<DataType>& data,
//                        const std::vector<size_t>& shape) {
//     std::shared_ptr<TensorImpl> impl =
//         std::make_shared<TensorImplModel<DataType, Backend>>(data, shape);
//     return Tensor(impl);
//   }

//   const size_t size() const { return impl_->data_size(); }
//   template <typename T>
//   std::span<const T> data() const {
//     return std::span<const T>(static_cast<const T*>(impl_->data_ptr()),
//                               impl_->data_size());
//   }
//   const std::vector<size_t>& shape() const { return impl_->shape(); }

//   template <typename T>
//   void fill(T item) {
//     any_type gen = item;
//     impl_->fill(gen);
//   }

//   friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
// };

}  // namespace autograd

#endif  // TENSOR_H