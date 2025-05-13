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

namespace lmp::tensor {

namespace detail {

class UnsafeTensorAccessor;

}

class Tensor {
 public:
  Tensor() = default;

  template <typename T>
  explicit Tensor(const std::vector<T>& data, const std::vector<size_t>& shape,
                  DeviceType device = DeviceType::CUDA,
                  DataType dtype = DataType::Float64)
      : impl_(std::make_shared<TensorImpl>(data, shape, device, dtype)) {}

  void* data() const { return impl_->data(); }
  DataType type() const { return impl_->type(); }
  DeviceType device() const { return impl_->device(); }
  const std::vector<size_t>& shape() const { return impl_->shape(); }
  size_t size() const { return impl_->size(); }

  // these functions only return an view
  template <typename T>
  std::span<T> view() const {
    static thread_local std::vector<T> converted_data;
    LMP_DISPATCH_ALL_TYPES(impl_->type(), [&] {
      // TODO: there's gotta be a better way to do this that's faster
      converted_data.resize(impl_->size());

      scalar_t* original_data =
          static_cast<scalar_t*>(malloc(size() * sizeof(scalar_t)));
      detail::native::copy_stub(device(), DeviceType::CPU, data(),
                                original_data, size(), type(), type());

      for (size_t i = 0; i < impl_->size(); ++i) {
        converted_data[i] = static_cast<T>(original_data[i]);
      }
      free(original_data);
    });
    return std::span<T>(converted_data);
  }
  Tensor reshape(std::vector<size_t> new_shape);
  Tensor squeeze(size_t dim);
  Tensor expand_dims(size_t dim);

  // these functions modify the actual data
  void copy(const Tensor& other);
  void fill(Scalar item);
  void to(DeviceType device);

  friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
  friend class TensorOpFact;
  friend class detail::UnsafeTensorAccessor;

 private:
  explicit Tensor(std::shared_ptr<TensorImpl> ptr) : impl_(ptr) {}
  std::shared_ptr<TensorImpl> impl_;
};

namespace detail {

struct UnsafeTensorAccessor {
  static std::shared_ptr<TensorImpl> getImpl(const Tensor& ten) {
    return ten.impl_;
  }
  static Tensor fromImpl(std::shared_ptr<TensorImpl> ptr) {
    return Tensor(ptr);
  }
};

}  // namespace detail

}  // namespace lmp::tensor