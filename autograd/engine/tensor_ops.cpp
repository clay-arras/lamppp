#include "tensor_ops.h"
#include <Eigen/Core>
#include <cassert>

namespace autograd {
inline namespace ops {

Tensor operator+(const Tensor& tensor, float scalar) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = tensor.as_array() + scalar;
  return Tensor(res_data, tensor.shape);
}

Tensor operator-(const Tensor& tensor, float scalar) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = tensor.as_array() - scalar;
  return Tensor(res_data, tensor.shape);
}

Tensor operator*(const Tensor& tensor, float scalar) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = tensor.as_array() * scalar;
  return Tensor(res_data, tensor.shape);
}

Tensor operator/(const Tensor& tensor, float scalar) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = tensor.as_array() / scalar;
  return Tensor(res_data, tensor.shape);
}

Tensor operator+(float scalar, const Tensor& tensor) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = scalar + tensor.as_array();
  return Tensor(res_data, tensor.shape);
}

Tensor operator-(float scalar, const Tensor& tensor) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = scalar - tensor.as_array();
  return Tensor(res_data, tensor.shape);
}

Tensor operator*(float scalar, const Tensor& tensor) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = scalar * tensor.as_array();
  return Tensor(res_data, tensor.shape);
}

Tensor operator/(float scalar, const Tensor& tensor) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = scalar / tensor.as_array();
  return Tensor(res_data, tensor.shape);
}

Tensor operator==(const Tensor& tensor, float scalar) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (tensor.as_array() == scalar).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator!=(const Tensor& tensor, float scalar) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (tensor.as_array() != scalar).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator>=(const Tensor& tensor, float scalar) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (tensor.as_array() >= scalar).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator<=(const Tensor& tensor, float scalar) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (tensor.as_array() <= scalar).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator>(const Tensor& tensor, float scalar) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (tensor.as_array() > scalar).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator<(const Tensor& tensor, float scalar) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (tensor.as_array() < scalar).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator==(float scalar, const Tensor& tensor) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (scalar == tensor.as_array()).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator!=(float scalar, const Tensor& tensor) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (scalar != tensor.as_array()).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator>=(float scalar, const Tensor& tensor) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (scalar >= tensor.as_array()).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator<=(float scalar, const Tensor& tensor) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (scalar <= tensor.as_array()).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator>(float scalar, const Tensor& tensor) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (scalar > tensor.as_array()).cast<float>();
  return Tensor(res_data, tensor.shape);
}

Tensor operator<(float scalar, const Tensor& tensor) {
  std::vector<float> res_data(tensor.data.size());
  Eigen::Map<Eigen::ArrayXf> res(res_data.data(), tensor.data.size());
  res = (scalar < tensor.as_array()).cast<float>();
  return Tensor(res_data, tensor.shape);
}

}  // namespace ops

}  // namespace autograd