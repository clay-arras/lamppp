#include "tensor.h"
#include <cassert>
#include <iostream>

namespace autograd {

Tensor Tensor::operator+(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = as_array() + other.as_array();
  return Tensor(res_data, shape);
}

Tensor Tensor::operator-(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = as_array() - other.as_array();
  return Tensor(res_data, shape);
}

Tensor Tensor::operator*(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = as_array() * other.as_array();
  return Tensor(res_data, shape);
}

Tensor Tensor::operator/(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = as_array() / other.as_array();
  return Tensor(res_data, shape);
}

Tensor Tensor::operator==(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() == other.as_array()).cast<float>();
  return Tensor(res_data, shape);
}

Tensor Tensor::operator!=(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() != other.as_array()).cast<float>();
  return Tensor(res_data, shape);
}

Tensor Tensor::operator>=(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() >= other.as_array()).cast<float>();
  return Tensor(res_data, shape);
}

Tensor Tensor::operator<=(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() <= other.as_array()).cast<float>();
  return Tensor(res_data, shape);
}

Tensor Tensor::operator>(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() > other.as_array()).cast<float>();
  return Tensor(res_data, shape);
}

Tensor Tensor::operator<(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() < other.as_array()).cast<float>();
  return Tensor(res_data, shape);
}

Tensor Tensor::matmul(const Tensor& other)
    const {  // TODO(nlin): maybe implement as static method that takes in A and B?
  assert(shape.size() == 2);
  assert(other.shape.size() == 2);
  assert(shape[1] == other.shape[0]);

  std::vector<float> res_data(shape[0] * other.shape[1]);
  Eigen::Map<Eigen::MatrixXf> res(res_data.data(), shape[0], other.shape[1]);

  res = as_matrix(shape[0], shape[1]) *
        other.as_matrix(other.shape[0], other.shape[1]);
  return Tensor(res_data, {shape[0], other.shape[1]});
}

Tensor Tensor::transpose() const {
  std::vector<float> res_data(data.size());
  std::vector<int> new_shape = {shape[1], shape[0]};

  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> res_matrix(
      res_data.data(), new_shape[0], new_shape[1]);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      src_matrix(data.data(), shape[0], shape[1]);

  res_matrix = src_matrix.transpose();
  return Tensor(res_data, new_shape);
}

Tensor Tensor::log() const {
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = as_array().log();
  return Tensor(res_data, shape);
}

Tensor Tensor::exp() const {
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = as_array().exp();
  return Tensor(res_data, shape);
}

Tensor Tensor::relu() const {
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = as_array().max(0.0F);
  return Tensor(res_data, shape);
}

Tensor Tensor::sum(int axis) const {
  assert(axis >= 0 && axis < static_cast<int>(shape.size()));
  std::vector<int> new_shape = shape;
  new_shape[axis] = 1;

  std::vector<float> res_data(data.size() / shape[axis]);
  if (axis == 0) {
    Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>> res(res_data.data(), 1,
                                                            shape[1]);
    res = as_array().reshaped(shape[0], shape[1]).colwise().sum().array();
  } else if (axis == 1) {
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> res(res_data.data(),
                                                            shape[0], 1);
    res = as_array().reshaped(shape[0], shape[1]).rowwise().sum().array();
  } else {
    assert(
        0);  // TODO(nlin): need to implement general thingy with collapsing!!!!
  } // TODO(nlin): this is disgusting please refactor ASAP
  return Tensor(res_data, new_shape);
}

Tensor Tensor::max(int axis) const {
  assert(axis >= 0 && axis < static_cast<int>(shape.size()));
  std::vector<int> new_shape = shape;
  new_shape[axis] = 1;

  std::vector<float> res_data(data.size() / shape[axis]);
  if (axis == 0) {
    Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>> res(res_data.data(), 1,
                                                            shape[1]);
    res = as_array().reshaped(shape[0], shape[1]).colwise().maxCoeff().array();
  } else if (axis == 1) {
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> res(res_data.data(),
                                                            shape[0], 1);
    res = as_array().reshaped(shape[0], shape[1]).rowwise().maxCoeff().array();
  } else {
    assert(
        0);  
  } 
  return Tensor(res_data, new_shape);
}

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
  os << "Tensor(data=[";
  for (size_t i = 0; i < obj.data.size(); i++) {
    os << obj.data[i];
    if (i < obj.data.size() - 1) {
      os << ", ";
    }
  }
  os << "], shape=[";
  for (size_t i = 0; i < obj.shape.size(); i++) {
    os << obj.shape[i];
    if (i < obj.shape.size() - 1) {
      os << ", ";
    }
  }
  os << "])";
  return os;
}

}  // namespace autograd