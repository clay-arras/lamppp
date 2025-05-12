#include "include/lamppp/tensor/tensor_helper.hpp"
#include "include/lamppp/tensor/scalar.hpp"

namespace lmp::tensor {

Tensor full_like(const Tensor& tensor, Scalar scalar) {
  std::vector<Scalar> data(tensor.size(), scalar);
  return Tensor(data, tensor.shape(), tensor.device(), tensor.type());
}

Tensor ones_like(const Tensor& tensor) {
  return full_like(tensor, 1);
}

Tensor zeros_like(const Tensor& tensor) {
  return full_like(tensor, 0);
}

}  // namespace lmp::tensor