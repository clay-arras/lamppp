#pragma once

#ifndef _TENSOR_H_
#define _TENSOR_H

#include "autograd/engine/backend/cuda_backend.hpp"
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
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

  template <typename T>
  Tensor(const std::vector<T>& data, const std::vector<size_t>& shape) {
    std::shared_ptr<AbstractBackend> cuda_backend = std::make_shared<CudaBackend>();
    impl_ = std::make_shared<TensorImpl>(Storage(data, shape, DataType::Float64), cuda_backend);
  }
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

}  // namespace autograd

#endif  // TENSOR_H