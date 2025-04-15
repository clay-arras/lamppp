#include <iostream>
#include "autograd/engine/tensor.h"
#include "autograd/engine/variable_ops.h"
#include "autograd/engine/variable.h"

namespace autograd {

namespace {

void test_backward_operations() {
  std::vector<float> x_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  std::vector<float> y_data = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
  std::vector<float> z_data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<int> shape = {2, 2, 2};

  Variable x(Tensor(x_data, shape), true);
  Variable y(Tensor(y_data, shape), true);
  Variable z(Tensor(z_data, shape), true);

  // Compute forward pass
  Variable a = x + y;
  Variable b = x - z;
  Variable c = y * z;
  Variable d = x / (y + 0.1F);
  Variable e = (x + 0.1F).log();
  Variable f = z.exp();

  Variable g = a * b + c;
  Variable h = d * e - f;

  Variable result = g + h * (e + 0.1F);

  result.backward();

  // Print gradients
  std::cout << "Gradient of x:" << std::endl;
  std::cout << x.grad() << std::endl;

  std::cout << "\nGradient of y:" << std::endl;
  std::cout << y.grad() << std::endl;

  std::cout << "\nGradient of z:" << std::endl;
  std::cout << z.grad() << std::endl;

}

}

}  // namespace

int main() {
  autograd::test_backward_operations();
  return 0;
}
