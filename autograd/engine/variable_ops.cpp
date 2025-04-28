#include "variable_ops.h"
#include "tensor.h"

namespace autograd {

inline namespace ops {

Variable operator+(const Variable& var, float scalar) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return var + Variable(scalar_tensor);
}

Variable operator-(const Variable& var, float scalar) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return var - Variable(scalar_tensor);
}

Variable operator*(const Variable& var, float scalar) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return var * Variable(scalar_tensor);
}

Variable operator/(const Variable& var, float scalar) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return var / Variable(scalar_tensor);
}

Variable operator+(float scalar, const Variable& var) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return Variable(scalar_tensor) + var;
}

Variable operator-(float scalar, const Variable& var) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return Variable(scalar_tensor) - var;
}

Variable operator*(float scalar, const Variable& var) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return Variable(scalar_tensor) * var;
}

Variable operator/(float scalar, const Variable& var) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return Variable(scalar_tensor) / var;
}

Variable operator==(const Variable& var, float scalar) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return var == Variable(scalar_tensor);
}

Variable operator!=(const Variable& var, float scalar) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return var != Variable(scalar_tensor);
}

Variable operator>=(const Variable& var, float scalar) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return var >= Variable(scalar_tensor);
}

Variable operator<=(const Variable& var, float scalar) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return var <= Variable(scalar_tensor);
}

Variable operator>(const Variable& var, float scalar) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return var > Variable(scalar_tensor);
}

Variable operator<(const Variable& var, float scalar) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return var < Variable(scalar_tensor);
}

Variable operator==(float scalar, const Variable& var) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return Variable(scalar_tensor) == var;
}

Variable operator!=(float scalar, const Variable& var) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return Variable(scalar_tensor) != var;
}

Variable operator>=(float scalar, const Variable& var) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return Variable(scalar_tensor) >= var;
}

Variable operator<=(float scalar, const Variable& var) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return Variable(scalar_tensor) <= var;
}

Variable operator>(float scalar, const Variable& var) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return Variable(scalar_tensor) > var;
}

Variable operator<(float scalar, const Variable& var) {
  Tensor scalar_tensor(var.data());
  scalar_tensor.fill(scalar);
  return Variable(scalar_tensor) < var;
}

}  // namespace ops

}  // namespace autograd