#pragma once

#include <iostream>
#include <memory>
#include <span>
#include <vector>
#include "data_type.hpp"
#include "device_type.hpp"
#include "dispatch_type.hpp"
#include "fill_like.hpp"
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
                  DeviceType device = DeviceType::CPU,
                  DataType dtype = DataType::Float64)
      : impl_(std::make_shared<TensorImpl>(data, shape, device, dtype)) {}

  void* data() const noexcept;
  DataType type() const noexcept;
  DeviceType device() const noexcept;
  const std::vector<size_t>& shape() const noexcept;
  const std::vector<detail::stride_t>& strides() const noexcept;
  size_t numel() const noexcept;

  // these functions only return an view
  template <typename T>
  std::vector<T> to_vector() const {
    std::vector<T> converted_data(impl_->numel());
    LMP_DISPATCH_ALL_TYPES(impl_->type(), [&] {
      scalar_t* original_data =
          static_cast<scalar_t*>(malloc(numel() * sizeof(scalar_t)));
      detail::native::copy_stub()(device(), DeviceType::CPU, data(),
                                  original_data, numel(), type(), type());

      for (size_t i = 0; i < impl_->numel(); ++i) {
        converted_data[i] = static_cast<T>(original_data[i]);
      }
      free(original_data);
    });
    return converted_data;
  }
  Tensor reshape(std::vector<size_t> new_shape) const;
  Tensor squeeze(size_t dim) const;
  Tensor expand_dims(size_t dim) const;
  Scalar index(const std::vector<size_t>& idx);

  // these functions modify the actual data
  void copy(const Tensor& other);
  void fill(Scalar item);
  Tensor to(DeviceType device);

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