#include "variable_ops.h"
#include "tensor.h"

namespace autograd {
inline namespace ops {

Variable operator+(const Variable& var, float scalar) {
  return var + Variable(Tensor(std::vector<float>(var.data().data.size(), scalar),
                       var.data().shape));
}

Variable operator-(const Variable& var, float scalar) {
  return var - Variable(Tensor(std::vector<float>(var.data().data.size(), scalar),
                       var.data().shape));
}

Variable operator*(const Variable& var, float scalar) {
  return var * Variable(Tensor(std::vector<float>(var.data().data.size(), scalar),
                       var.data().shape));
}

Variable operator/(const Variable& var, float scalar) {
  return var / Variable(Tensor(std::vector<float>(var.data().data.size(), scalar),
                       var.data().shape));
}

Variable operator+(float scalar, const Variable& var) {
  Variable scalar_var(Tensor(std::vector<float>(var.data().data.size(), scalar),
                    var.data().shape));
  return scalar_var.operator+(var);
}

Variable operator-(float scalar, const Variable& var) {
  return Variable(Tensor(std::vector<float>(var.data().data.size(), scalar),
                 var.data().shape)) - var;
}

Variable operator*(float scalar, const Variable& var) {
  Variable scalar_var(Tensor(std::vector<float>(var.data().data.size(), scalar),
                    var.data().shape));
  return scalar_var.operator*(var);
}

Variable operator/(float scalar, const Variable& var) {
  return Variable(Tensor(std::vector<float>(var.data().data.size(), scalar),
                 var.data().shape)) / var;
}

bool operator==(const Variable& lhs, const Variable& rhs) {
  return lhs.data() == rhs.data();
}

} // namespace ops
} // namespace autograd 