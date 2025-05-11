#pragma once

#include <iostream>
#include <memory>
#include <span>
#include <vector>
#include "data_type.hpp"
#include "device_type.hpp"
#include "dispatch_type.hpp"
#include "tensor_helper.hpp"
#include "tensor_impl.hpp"

namespace autograd {

class Tensor {
 public:
  // Tensor() = default;  // TODO: is this really necessary, remove if can
  Tensor() {
    impl_ = std::make_shared<TensorImpl>();
  };  // TODO: is this really necessary, remove if can
  Tensor(const Tensor& other) : impl_(other.impl_) {
    // std::cout << "COPY CONSTRUCTOR: " << other << std::endl;
  }

  Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {
    // std::cout << "MOVE CONSTRUCTOR: " << other << std::endl;
  }

  Tensor& operator=(const Tensor& other) {
    // std::cout << "BEFORE COPY ASSIGNMENT: " << other << std::endl;
    // std::cout << "BEFORE COPY ASSIGNMENT THIS: " << *this << std::endl;
    if (this != &other) {
      impl_ = other.impl_;
      // std::cout << "AFTER COPY ASSIGNMENT: " << other << std::endl;
      // std::cout << "AFTER COPY ASSIGNMENT THIS: " << *this << std::endl;
    }
    return *this;
  }

  template <typename T>  // TODO: maybe make this implicit?
  explicit Tensor(const std::vector<T>& data, const std::vector<size_t>& shape,
                  DeviceType device = DeviceType::CUDA,
                  DataType dtype = DataType::Float64)
      : impl_(std::make_shared<TensorImpl>(data, shape, device, dtype)) {}

  void* data() const { return impl_->data(); }
  DataType type() const { return impl_->type(); }
  DeviceType device() const { return impl_->device(); }
  const std::vector<size_t>& shape() const { return impl_->shape(); }
  size_t size() const { return impl_->size(); }

  template <typename T>
  std::span<T> view() const {
    static thread_local std::vector<T> converted_data;
    DISPATCH_ALL_TYPES(impl_->type(), [&] {
      // TODO: there's gotta be a better way to do this that's faster
      converted_data.resize(impl_->size());

      scalar_t* original_data =
          static_cast<scalar_t*>(malloc(size() * sizeof(scalar_t)));
      copy_stub(device(), DeviceType::CPU, data(), original_data, size(),
                type(), type());

      for (size_t i = 0; i < impl_->size(); ++i) {
        converted_data[i] = static_cast<T>(original_data[i]);
      }
      free(original_data);
    });
    return std::span<T>(converted_data);
  }

  void copy(const Tensor& other);
  void fill(Scalar item);
  void to(DeviceType device);

  friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);

  inline friend Tensor log(const Tensor& tensor) {
    auto result = TensorImpl::log(*tensor.impl_);
    return Tensor(std::make_shared<TensorImpl>(result));
  }
  inline friend Tensor exp(const Tensor& tensor) {
    auto result = TensorImpl::exp(*tensor.impl_);
    return Tensor(std::make_shared<TensorImpl>(result));
  }
  inline friend Tensor relu(const Tensor& tensor) {
    auto result = TensorImpl::relu(*tensor.impl_);
    return Tensor(std::make_shared<TensorImpl>(result));
  }
  inline friend Tensor matmul(const Tensor& a, const Tensor& b) {
    auto result = TensorImpl::matmul(*a.impl_, *b.impl_);
    return Tensor(std::make_shared<TensorImpl>(result));
  }
  inline friend Tensor transpose(const Tensor& tensor) {
    auto result = TensorImpl::transpose(*tensor.impl_);
    return Tensor(std::make_shared<TensorImpl>(result));
  }
  inline friend Tensor sum(const Tensor& tensor, size_t axis) {
    auto result = TensorImpl::sum(*tensor.impl_, axis);
    return Tensor(std::make_shared<TensorImpl>(result));
  }
  inline friend Tensor max(const Tensor& tensor, size_t axis) {
    auto result = TensorImpl::max(*tensor.impl_, axis);
    return Tensor(std::make_shared<TensorImpl>(result));
  }

  template <auto OpTag>
  static inline Tensor binary_tensor_op(const Tensor& a, const Tensor& b) {
    auto result = (*OpTag)(*a.impl_, *b.impl_);
    Tensor tmp(std::make_shared<TensorImpl>(result));
    // std::cout << "INTERM TMP result " << tmp << std::endl;
    return tmp;
  }

  template <auto OpTag>
  static inline Tensor binary_tensor_op(const Tensor& tensor, Scalar scalar) {
    Tensor scalar_tensor(std::vector<Scalar>(tensor.size(), scalar),
                         tensor.shape(), tensor.device(), tensor.type());
    return binary_tensor_op<OpTag>(tensor, scalar_tensor);
  }

  template <auto OpTag>
  static inline Tensor binary_tensor_op(Scalar scalar, const Tensor& tensor) {
    Tensor scalar_tensor(std::vector<Scalar>(tensor.size(), scalar),
                         tensor.shape(), tensor.device(), tensor.type());
    return binary_tensor_op<OpTag>(scalar_tensor, tensor);
  }

#define DECL_BINARY_OP(op, tag)                                           \
  inline friend Tensor operator op(const Tensor& a, const Tensor& b) {    \
    return binary_tensor_op<&TensorImpl::tag>(a, b);                      \
  }                                                                       \
  inline friend Tensor operator op(const Tensor& tensor, Scalar scalar) { \
    return binary_tensor_op<&TensorImpl::tag>(tensor, scalar);            \
  }                                                                       \
  inline friend Tensor operator op(Scalar scalar, const Tensor& tensor) { \
    return binary_tensor_op<&TensorImpl::tag>(scalar, tensor);            \
  }

#define FORALL_BINARY_OPS(_) \
  _(+, add)                  \
  _(-, sub)                  \
  _(*, mul)                  \
  _(/, div)                  \
  _(==, equal)               \
  _(!=, not_equal)           \
  _(>=, greater_equal)       \
  _(<=, less_equal)          \
  _(>, greater_than)         \
  _(<, less_than)

  FORALL_BINARY_OPS(DECL_BINARY_OP)

#undef FORALL_BINARY_OPS
#undef DECL_BINARY_OP

 private:
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
  std::shared_ptr<TensorImpl> impl_;
};

}  // namespace autograd