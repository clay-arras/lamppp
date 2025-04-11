#include <iostream>
#include "autograd/engine/tensor.h"
#include "autograd/engine/variable.h"

namespace {

void test_tensor_operations() {
  Tensor a({1.0, 2.0, 3.0, 4.0}, {2, 2});  // 2x2 tensor
  Tensor b({5.0, 6.0, 7.0, 8.0}, {2, 2});  // 2x2 tensor

  Tensor c = a + b;
  std::cout << "a + b: ";
  for (const auto& val : c.data)
    std::cout << val << " ";
  std::cout << std::endl;

  Tensor d = a - b;
  std::cout << "a - b: ";
  for (const auto& val : d.data)
    std::cout << val << " ";
  std::cout << std::endl;

  Tensor e = a * b;
  std::cout << "a * b: ";
  for (const auto& val : e.data)
    std::cout << val << " ";
  std::cout << std::endl;

  Tensor f = a.matmul(b);
  std::cout << "a matmul b: ";
  for (const auto& val : f.data)
    std::cout << val << " ";
  std::cout << std::endl;

  Tensor g = a.log();
  std::cout << "log(a): ";
  for (const auto& val : g.data)
    std::cout << val << " ";
  std::cout << std::endl;

  Tensor h = a.relu();
  std::cout << "ReLU(a): ";
  for (const auto& val : h.data)
    std::cout << val << " ";
}

void test_backward_operations() {
  std::vector<float> data_a = {1.0, 2.0, 3.0, 4.0};
  std::vector<int> shape_a = {2, 2};
  Tensor a(data_a, shape_a);

  std::vector<float> data_b = {5.0, 6.0, 7.0, 8.0};
  std::vector<int> shape_b = {2, 2};
  Tensor b(data_b, shape_b);

  std::shared_ptr<VariableImpl> var_impl_a =
      std::make_shared<VariableImpl>(a, true);  // requires gradient
  Variable var_a = Variable(var_impl_a);
  std::shared_ptr<VariableImpl> var_impl_b =
      std::make_shared<VariableImpl>(b, true);  // requires gradient
  Variable var_b = Variable(var_impl_b);

  Variable var_c = var_a + var_b;  // a + b
  Variable var_d = var_a * var_b;  // a * b

  var_c.backward();
  var_d.backward();

  std::cout << "Gradients for a after a + b: " << var_a.grad().data[0] << ", "
            << var_a.grad().data[1] << std::endl;  // Expected: [1, 1]
  std::cout << "Gradients for b after a + b: " << var_b.grad().data[0] << ", "
            << var_b.grad().data[1] << std::endl;  // Expected: [1, 1]

  std::cout << "Gradients for a after a * b: " << var_a.grad().data[0] << ", "
            << var_a.grad().data[1] << std::endl;  // Expected: [b, b]
  std::cout << "Gradients for b after a * b: " << var_b.grad().data[0] << ", "
            << var_b.grad().data[1] << std::endl;  // Expected: [a, a]
}

}  // namespace

int main() {
  test_tensor_operations();
  test_backward_operations();
  return 0;
}
