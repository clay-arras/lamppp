#include "tensor_ops.h"
#include <cassert>

namespace autograd {
inline namespace ops {

Tensor operator+(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return tensor + scalar_tensor;
}

Tensor operator-(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return tensor - scalar_tensor;
}

Tensor operator*(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return tensor * scalar_tensor;
}

Tensor operator/(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return tensor / scalar_tensor;
}

Tensor operator+(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return scalar_tensor + tensor;
}

Tensor operator-(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return scalar_tensor - tensor;
}

Tensor operator*(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return scalar_tensor * tensor;
}

Tensor operator/(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return scalar_tensor / tensor;
}

Tensor operator==(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return tensor == scalar_tensor;
}

Tensor operator!=(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return tensor != scalar_tensor;
}

Tensor operator>=(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return tensor >= scalar_tensor;
}

Tensor operator<=(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return tensor <= scalar_tensor;
}

Tensor operator>(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return tensor > scalar_tensor;
}

Tensor operator<(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return tensor < scalar_tensor;
}

Tensor operator==(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return scalar_tensor == tensor;
}

Tensor operator!=(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return scalar_tensor != tensor;
}

Tensor operator>=(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return scalar_tensor >= tensor;
}

Tensor operator<=(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return scalar_tensor <= tensor;
}

Tensor operator>(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return scalar_tensor > tensor;
}

Tensor operator<(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(std::vector<float>(tensor.size(), scalar), tensor.shape());
  return scalar_tensor < tensor;
}

}  // namespace ops

}  // namespace autograd