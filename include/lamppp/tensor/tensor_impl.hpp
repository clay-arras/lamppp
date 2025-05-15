#pragma once

#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
#include "data_type.hpp"
#include "device_type.hpp"
#include "dispatch_type.hpp"
#include "include/lamppp/tensor/align_utils.hpp"
#include "include/lamppp/tensor/native/copy.cuh"
#include "include/lamppp/tensor/storage.hpp"
#include "scalar.hpp"

namespace lmp::tensor {

class TensorImpl {
 public:
  template <typename T>
  explicit TensorImpl(const std::vector<T>& data,
                      const std::vector<size_t>& shape, DeviceType device,
                      DataType dtype)
      : data_(LMP_DISPATCH_ALL_TYPES(
            dtype,
            [&] { return Storage(data.size() * sizeof(scalar_t), device); })),
        shape_(shape),
        type_(dtype),
        strides_(std::vector<detail::stride_t>(shape.size())),
        size_(shape.empty() ? 0
                            : std::accumulate(shape.begin(), shape.end(), 1,
                                              std::multiplies<>())) {
    assert(data.size() == size_ &&
           "Size mismatch, product of shape must equal num elements");
    DataType src_dtype = TypeMeta<T>::value;
    detail::native::copy_stub(DeviceType::CPU, device, data.data(),
                              data_.data(), size_, src_dtype, type_);
    update_strides_();
  }
  explicit TensorImpl(const Storage& storage, const std::vector<size_t>& shape,
                      DataType dtype);

  void* data() const;
  DataType type() const;
  DeviceType device() const;
  const std::vector<size_t>& shape() const;
  const std::vector<detail::stride_t>& strides() const;
  size_t size() const;

  TensorImpl reshape_(std::vector<size_t> new_shape);
  TensorImpl squeeze_(size_t dim);
  TensorImpl expand_dims_(size_t dim);

  void copy_(const TensorImpl& other);
  void fill_(Scalar item);
  void to_(DeviceType device);
  void print_(std::ostream& os);

 private:
  friend class Tensor;
  void update_strides_();

  DataType type_;
  Storage data_;
  size_t size_;
  std::vector<size_t> shape_;
  std::vector<detail::stride_t> strides_;
};

}  // namespace lmp::tensor