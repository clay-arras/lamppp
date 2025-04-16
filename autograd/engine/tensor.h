#pragma once
#ifndef _TENSOR_H_
#define _TENSOR_H

#include <Eigen/Core>
#include <vector>

namespace autograd {

class Tensor {
 public:
  std::vector<float> data;
  std::vector<int> shape;

  Tensor() = default;
  Tensor(const std::vector<float>& data, const std::vector<int>& shape)
      : data(data), shape(shape) {}

  const int size() const;
  
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

  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_matrix(int rows, int cols) const { // TODO(nlin): make it work with default Eigen column major order
    return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      const_cast<float*>(data.data()), rows, cols);
  }

  // Eigen::Map<Eigen::MatrixXf> as_matrix(int rows, int cols) const {
  //   return Eigen::Map<Eigen::MatrixXf>(const_cast<float*>(data.data()), cols, rows);
  // }

  Eigen::Map<Eigen::ArrayXf> as_array() const {
    return Eigen::Map<Eigen::ArrayXf>(const_cast<float*>(data.data()),
                                     data.size());
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
};

}

#endif