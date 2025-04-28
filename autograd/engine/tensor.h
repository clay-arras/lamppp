#pragma once

#ifndef _TENSOR_H_
#define _TENSOR_H

#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <span>
#include <vector>
#include "tensor_impl.h"

namespace autograd {

class Tensor {
 public:
  // template<typename DataType, typename Backend>
  // explicit Tensor(const std::vector<DataType>& data, const std::vector<int>& shape)
  //     : impl_(std::make_shared<TensorImplModel<DataType, Backend>>(data, shape)) {}
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

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
  const std::span<T>& data() const {
    return std::span<T>(impl_->data_ptr());
  }
  const std::vector<int>& shape() const { return impl_->shape(); }

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

// struct TensorOpFact {
//   template <typename EigenOpFn, typename... OtherTensors>
//   static Tensor apply(const EigenOpFn& op_fn, const std::vector<int>& out_shape,
//                       const Tensor& tensor,
//                       const OtherTensors&... other_tensors) {
//     int sz = std::accumulate(out_shape.begin(), out_shape.end(), 1,
//                              std::multiplies<>());
//     assert(sz == out_shape[0] * out_shape[1]);
//     std::vector<float> res_data(sz);
//     Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//     res = op_fn(tensor, other_tensors...).reshaped(sz, 1);
//     return Tensor(res_data, out_shape);
//   }
// };

}  // namespace autograd

#endif  // TENSOR_H