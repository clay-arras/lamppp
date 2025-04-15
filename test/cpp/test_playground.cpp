#include <iostream>
#include <unordered_set>
#include "autograd/engine/tensor.h"
#include "autograd/engine/variable.h"

namespace {
using std::make_shared;

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
  std::cout << std::endl;
}

void test_backward_operations() {
  Tensor a({1.0, 2.0, -1.0, 0.0}, {2, 2});
  std::shared_ptr<VariableImpl> var_impl_a = make_shared<VariableImpl>(a, true);  // requires gradient
  Variable var_a = Variable(var_impl_a);

  Tensor b({7.0, 6.0, -9.0, 2.0}, {2, 2});
  std::shared_ptr<VariableImpl> var_impl_b = make_shared<VariableImpl>(b, true);  // requires gradient
  Variable var_b = Variable(var_impl_b);

  Tensor c({9.0, 10.0, -1.0, -1.0}, {2, 2});
  std::shared_ptr<VariableImpl> var_impl_c = make_shared<VariableImpl>(c, true);  // requires gradient
  Variable var_c = Variable(var_impl_c);

  Variable var_d = var_a*var_b + var_c;
  var_d.backward();

  std::cout << "Var_A: " << var_a << std::endl;
  std::cout << "Var_B: " << var_b << std::endl;
  std::cout << "Var_C: " << var_c << std::endl;
}

}  // namespace

int main() {
  std::unordered_set<Variable> x;

  std::vector<float> data_b = {5.0, 6.0, 7.0, 8.0};
  std::vector<int> shape_b = {2, 2};
  Tensor b(data_b, shape_b);

  std::shared_ptr<VariableImpl> var_impl_b =
      std::make_shared<VariableImpl>(b, true);  // requires gradient
  Variable var_a = Variable(var_impl_b);
  assert(x.find(var_a) == x.end());

  // test_tensor_operations();
  test_backward_operations();
  return 0;
}
