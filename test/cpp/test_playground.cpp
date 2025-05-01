#include <vector>
// #include "autograd/autograd_umbrella.h"
#include "autograd/engine/backend/cuda_backend.hpp"
#include "autograd/engine/tensor.hpp"
#include "autograd/engine/variable.hpp"
#include "autograd/engine/variable_ops.hpp"

int main() {
  std::vector<int> data1 = {1, 2, 3, 4, 5, 6};
  std::vector<size_t> shape1 = {3, 2};
  autograd::Tensor tensor1 =
      autograd::Tensor::create<int, autograd::CudaBackend<int>>(data1, shape1);
  autograd::Variable var1(tensor1, true);

  std::vector<int> data2 = {7, 8, 9, 10, 11, 12};
  autograd::Tensor tensor2 =
      autograd::Tensor::create<int, autograd::CudaBackend<int>>(data2, shape1);
  autograd::Variable var2(tensor2, true);

  autograd::Variable result = var1 + var2;

  std::cout << "Variable 1: " << var1 << std::endl;
  std::cout << "Variable 2: " << var2 << std::endl;
  std::cout << "Result of tensor addition: " << result << std::endl;

  // result.impl_->grad.fill(1);
  result.backward();

  std::cout << std::endl;
  std::cout << "Variable 1: " << var1 << std::endl;
  std::cout << "Variable 2: " << var2 << std::endl;
  std::cout << "Result of tensor addition: " << result << std::endl;

  // std::cout << var1 << std::endl;
  // var1.impl_->grad.fill(1);
  // std::cout << var1 << std::endl;

  return 0;
}
