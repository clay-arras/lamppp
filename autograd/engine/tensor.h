#pragma once

#ifndef _TENSOR_H_
#define _TENSOR_H

#include <Eigen/Core>
#include <numeric>
#include <vector>

namespace autograd {

class Tensor {
 public:
  std::vector<float> data;
  std::vector<int> shape;
  int size() const { return data.size(); };

  Tensor() = default;
  Tensor(const std::vector<float>& data, const std::vector<int>& shape)
      : data(data), shape(shape) {}

  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  Tensor operator==(const Tensor& other) const;
  Tensor operator!=(const Tensor& other) const;
  Tensor operator>=(const Tensor& other) const;
  Tensor operator<=(const Tensor& other) const;
  Tensor operator>(const Tensor& other) const;
  Tensor operator<(const Tensor& other) const;

  Tensor matmul(const Tensor& other) const;
  Tensor transpose() const;

  Tensor log() const;
  Tensor exp() const;
  Tensor relu() const;

  Tensor sum(int axis) const;
  Tensor mean(int axis) const;
  Tensor max(int axis) const;
  Tensor min(int axis) const;

  Eigen::Map<Eigen::MatrixXf> as_matrix(int rows, int cols) const {
    return Eigen::Map<Eigen::MatrixXf>(const_cast<float*>(data.data()), rows,
                                       cols);
  }

  Eigen::Map<Eigen::ArrayXXf> as_array() const {
    return Eigen::Map<Eigen::ArrayXXf>(const_cast<float*>(data.data()),
                                      data.size(), 1);
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
};

struct TensorOpFact {
  template <typename EigenOpFn, typename... OtherTensors>
  static Tensor apply(const EigenOpFn& op_fn, const std::vector<int>& out_shape,
                      const Tensor& tensor,
                      const OtherTensors&... other_tensors) {
    int sz = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                             std::multiplies<>());
    assert(sz == out_shape[0] * out_shape[1]);
    std::vector<float> res_data(sz);
    Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
    res = op_fn(tensor, other_tensors...).reshaped(sz, 1);
    return Tensor(res_data, out_shape);
  }
};

}  // namespace autograd

#endif