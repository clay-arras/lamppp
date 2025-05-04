#pragma once

#ifndef _TENSOR_H_
#define _TENSOR_H

#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <span>
#include <vector>
#include "autograd/engine/backend.hpp"
#include "autograd/engine/backend/cuda_backend.hpp"
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/storage.hpp"
#include "dispatch.hpp"
#include "tensor_impl.hpp"

namespace autograd {

class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

  template <typename T>
  Tensor(const std::vector<T>& data, const std::vector<size_t>& shape) {
    std::shared_ptr<AbstractBackend> cuda_backend =
        std::make_shared<CudaBackend>();
    impl_ = std::make_shared<TensorImpl>(
        Storage(data, shape, DataType::Float64), cuda_backend);
  }

  void* data() const { return impl_->data(); }
  DataType type() const { return impl_->type(); }
  const std::vector<size_t>& shape() const { return impl_->shape(); }
  size_t size() const { return impl_->size(); }
  std::shared_ptr<AbstractBackend> backend() const { return impl_->backend(); }

  template <typename T>
  static Tensor create(const std::vector<T>& data,
                       const std::vector<size_t>& shape,
                       std::shared_ptr<AbstractBackend> backend,
                       DataType dtype) {
    std::shared_ptr<TensorImpl> impl =
        std::make_shared<TensorImpl>(Storage(data, shape, dtype), backend);
    return Tensor(impl);
  }

  void fill(Scalar item) { impl_->fill(item); }

  template <typename T>
  std::span<T> data() const {
    static thread_local std::vector<T> converted_data;
    DISPATCH_ALL_TYPES(impl_->type(), [&] {
      converted_data.resize(impl_->size());
      scalar_t* original_data = static_cast<scalar_t*>(impl_->data());
      for (size_t i = 0; i < impl_->size(); ++i) {
        converted_data[i] = static_cast<T>(original_data[i]);
      }
    });
    return std::span<T>(converted_data);
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);

  inline friend Tensor log(const Tensor& tensor) {
    auto result = TensorImpl::log(*tensor.impl_);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
  }
  inline friend Tensor exp(const Tensor& tensor) {
    auto result = TensorImpl::exp(*tensor.impl_);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
  }
  inline friend Tensor relu(const Tensor& tensor) {
    auto result = TensorImpl::relu(*tensor.impl_);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
  }
  inline friend Tensor matmul(const Tensor& a, const Tensor& b) {
    auto result = TensorImpl::matmul(*a.impl_, *b.impl_);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
  }
  inline friend Tensor transpose(const Tensor& tensor) {
    auto result = TensorImpl::transpose(*tensor.impl_);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
  }
  inline friend Tensor sum(const Tensor& tensor, size_t axis) {
    auto result = TensorImpl::sum(*tensor.impl_, axis);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
  }
  inline friend Tensor max(const Tensor& tensor, size_t axis) {
    auto result = TensorImpl::max(*tensor.impl_, axis);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
  }

  template <auto OpTag>
  static inline Tensor binary_tensor_op(const Tensor& a, const Tensor& b) {
    auto result = (*OpTag)(*a.impl_, *b.impl_);
    return Tensor(std::make_shared<TensorImpl>(result));
  }

  template <auto OpTag>
  static inline Tensor binary_tensor_op(const Tensor& tensor, float scalar) {
    Tensor scalar_tensor = Tensor::create(
        std::vector<float>(tensor.size(), static_cast<float>(scalar)),
        tensor.shape(), tensor.backend(), tensor.type());
    return binary_tensor_op<OpTag>(tensor, scalar_tensor);
  }

  template <auto OpTag>
  static inline Tensor binary_tensor_op(float scalar, const Tensor& tensor) {
    Tensor scalar_tensor = Tensor::create(
        std::vector<float>(tensor.size(), static_cast<float>(scalar)),
        tensor.shape(), tensor.backend(), tensor.type());
    return binary_tensor_op<OpTag>(scalar_tensor, tensor);
  }

#define DECL_BINARY_OP(op, tag)                                   \
  inline friend Tensor operator op(const Tensor& a, const Tensor& b) {   \
    return binary_tensor_op<&TensorImpl::tag>(a, b);              \
  }                                                               \
  inline friend Tensor operator op(const Tensor& tensor, float scalar) { \
    return binary_tensor_op<&TensorImpl::tag>(tensor, scalar);    \
  }                                                               \
  inline friend Tensor operator op(float scalar, const Tensor& tensor) { \
    return binary_tensor_op<&TensorImpl::tag>(scalar, tensor);    \
  }

#define FORALL_BINARY_OPS(_) \
  _(+,add)                  \
  _(-,sub)                  \
  _(*,mul)                  \
  _(/,div)                  \
  _(==,equal)               \
  _(!=,not_equal)           \
  _(>=,greater_equal)       \
  _(<=,less_equal)          \
  _(>,greater_than)         \
  _(<,less_than)

FORALL_BINARY_OPS(DECL_BINARY_OP)

#undef FORALL_BINARY_OPS
#undef DECL_BINARY_OP

private: 
  std::shared_ptr<TensorImpl> impl_;
};

}  // namespace autograd

#endif  // TENSOR_H