#include "variable_ops.hpp"
#include "autograd/engine/functions/basic_ops.hpp"
#include "autograd/engine/functions/matrix_ops.hpp"
#include "autograd/engine/functions/reduct_ops.hpp"
#include "autograd/engine/functions/unary_ops.hpp"
#include "tensor.hpp"

namespace autograd {

inline namespace ops { // TODO: once we implement broadcasting, we can use implicit convertors to convert the scalar to a 1x1 variable, then inside the binary ops we use broadcasting

Variable operator+(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Add>({a, b})[0];
}

Variable operator-(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Subtract>({a, b})[0];
}

Variable operator*(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Multiply>({a, b})[0];
}

Variable operator/(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<Divide>({a, b})[0];
}

Variable exp(const Variable& a) {
  return VariableOpFact::apply<Exponential>({a})[0];
}

Variable log(const Variable& a) {
  return VariableOpFact::apply<Logarithm>({a})[0];
}

Variable relu(const Variable& a) {
  return VariableOpFact::apply<ReLU>({a})[0];
}

Variable matmul(const Variable& a, const Variable& b) {
  return VariableOpFact::apply<MatrixMultiplication>({a, b})[0];
}

Variable transpose(const Variable& a) {
  return VariableOpFact::apply<Transpose>({a})[0];
}

Variable sum(const Variable& a, int axis) {
  return VariableOpFact::apply<Summation>({a}, axis)[0];
}

Variable max(const Variable& a, int axis) {
  return VariableOpFact::apply<Maximum>({a}, axis)[0];
}

Variable operator==(const Variable& a, const Variable& b) {
  return Variable(a.data() == b.data(), false);
}

Variable operator!=(const Variable& a, const Variable& b) {
  return Variable(a.data() != b.data(), false);
}

Variable operator>=(const Variable& a, const Variable& b) {
  return Variable(a.data() >= b.data(), false);
}

Variable operator<=(const Variable& a, const Variable& b) {
  return Variable(a.data() <= b.data(), false);
}

Variable operator>(const Variable& a, const Variable& b) {
  return Variable(a.data() > b.data(), false);
}

Variable operator<(const Variable& a, const Variable& b) {
  return Variable(a.data() < b.data(), false);
}

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