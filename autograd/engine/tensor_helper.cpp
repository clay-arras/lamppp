#include "tensor_helper.hpp"
#include "autograd/engine/scalar.hpp"

namespace autograd {

Tensor full_like(const Tensor& tensor, Scalar scalar) {
  // std::cout << "IN FULL LIKE: " << tensor << std::endl;
  std::vector<Scalar> data(tensor.size(), scalar);
  return Tensor(data, tensor.shape(), tensor.device(), tensor.type());
}

Tensor ones_like(const Tensor& tensor) {
  return full_like(tensor, 1);
}

Tensor zeros_like(const Tensor& tensor) {
  return full_like(tensor, 0);
}

}  // namespace autograd