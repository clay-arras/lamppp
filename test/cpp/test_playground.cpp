#include <iostream>
#include "autograd/engine/backend/cuda_backend.hpp"
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/tensor.hpp"
#include "autograd/engine/tensor_helper.hpp"
#include "autograd/engine/variable.hpp"
#include "autograd/engine/variable_ops.hpp"

int main() {
  std::vector<float> data1 = {1.0f, 2.0f, -1.0f};
  std::vector<size_t> shape1 = {1, 3};
  autograd::Tensor tensor_data1 = autograd::Tensor(
      data1, shape1, std::make_shared<autograd::CudaBackend>(),
      DataType::Float32);

  std::vector<float> data2 = {1.0f, 2.0f, 3.0f};
  std::vector<size_t> shape2 = {1, 3};
  autograd::Tensor tensor_data2 = autograd::Tensor(
      data2, shape2, std::make_shared<autograd::CudaBackend>(),
      DataType::Float32);

  autograd::Variable variable_data1(tensor_data1, true);
  autograd::Variable variable_data2(tensor_data2, true);

  autograd::Variable result = variable_data1 + variable_data2;

//   std::cout << "Variable 1: " << variable_data1 << std::endl;
//   autograd::Variable variable_data2(variable_data1);
//   // variable_data2.grad().fill(1);

// std::cout << std::endl;
//   std::cout << "Variable 1: " << variable_data1 << std::endl;
//   std::cout << "Variable 2: " << variable_data2 << std::endl;

//   variable_data2.incr_grad(tensor_data2);

// std::cout << std::endl;
  std::cout << "Variable 1: " << variable_data1 << std::endl;
  std::cout << "Variable 2: " << variable_data2 << std::endl;
  std::cout << "Result: " << result << std::endl;

  std::cout << std::endl;

  result.backward();

  std::cout << "Variable 1: " << variable_data1 << std::endl;
  std::cout << "Variable 2: " << variable_data2 << std::endl;
  std::cout << "Result: " << result << std::endl;
}