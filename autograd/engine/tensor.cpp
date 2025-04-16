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
  res = (as_array() == other.as_array());
  return Tensor(res_data, shape);
}

Tensor Tensor::operator!=(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() != other.as_array());
  return Tensor(res_data, shape);
}

Tensor Tensor::operator>=(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() >= other.as_array());
  return Tensor(res_data, shape);
}

Tensor Tensor::operator<=(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() <= other.as_array());
  return Tensor(res_data, shape);
}

Tensor Tensor::operator>(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() > other.as_array());
  return Tensor(res_data, shape);
}

Tensor Tensor::operator<(const Tensor& other) const {
  assert(shape == other.shape);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array() < other.as_array());
  return Tensor(res_data, shape);
}

Tensor Tensor::matmul(const Tensor& other)
    const {  // TODO(nlin): maybe implement as static method that takes in A and B?
  assert(shape.size() == 2);
  assert(other.shape.size() == 2);
  assert(shape[1] == other.shape[0]);
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::MatrixXf> res(res_data.data(), shape[0], other.shape[1]);
  res = as_matrix(shape[0], shape[1]) *
        other.as_matrix(other.shape[0], other.shape[1]);
  return Tensor(res_data, shape);
}

Tensor Tensor::T() const {
  std::vector<float> res_data(data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), data.size());
  res = (as_array().transpose());
  return Tensor(res_data, shape);
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

}