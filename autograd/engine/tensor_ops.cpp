#include "tensor_ops.h"
#include <cassert>

namespace autograd {
inline namespace ops {

// TODO: need to generalize, or just get rid of later
Tensor operator+(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return tensor + scalar_tensor;
}

Tensor operator-(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return tensor - scalar_tensor;
}

Tensor operator*(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return tensor * scalar_tensor;
}

Tensor operator/(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return tensor / scalar_tensor;
}

Tensor operator+(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return scalar_tensor + tensor;
}

Tensor operator-(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return scalar_tensor - tensor;
}

Tensor operator*(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return scalar_tensor * tensor;
}

Tensor operator/(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return scalar_tensor / tensor;
}

Tensor operator==(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return tensor == scalar_tensor;
}

Tensor operator!=(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return tensor != scalar_tensor;
}

Tensor operator>=(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return tensor >= scalar_tensor;
}

Tensor operator<=(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return tensor <= scalar_tensor;
}

Tensor operator>(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return tensor > scalar_tensor;
}

Tensor operator<(const Tensor& tensor, float scalar) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return tensor < scalar_tensor;
}

Tensor operator==(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return scalar_tensor == tensor;
}

Tensor operator!=(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return scalar_tensor != tensor;
}

Tensor operator>=(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return scalar_tensor >= tensor;
}

Tensor operator<=(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return scalar_tensor <= tensor;
}

Tensor operator>(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return scalar_tensor > tensor;
}

Tensor operator<(float scalar, const Tensor& tensor) {
  Tensor scalar_tensor(tensor);
  scalar_tensor.fill(scalar);
  return scalar_tensor < tensor;
}

}  // namespace ops

}  // namespace autograd