#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "../autograd/engine.h"

bool approx_equal(double a, double b, double epsilon = 1e-2) {
  return std::abs(a - b) < epsilon;
}

int main() {
  auto x1 = std::make_shared<Value>(2.0);
  auto w1 = std::make_shared<Value>(-3.0);
  auto x2 = std::make_shared<Value>(0.0);
  auto w2 = std::make_shared<Value>(1.0);
  auto b = std::make_shared<Value>(6.8813735870195432);

  auto w1x1 = w1 * x1;
  auto w2x2 = w2 * x2;
  auto w1x1w2x2 = w1x1 + w2x2;
  auto z = w1x1w2x2 + b;
  auto a = z->tanh();

  a->backprop();
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "Forward pass values:" << std::endl;
  std::cout << "w1*x1 = " << w1x1->data << std::endl;
  std::cout << "w2*x2 = " << w2x2->data << std::endl;
  std::cout << "z = " << z->data << std::endl;
  std::cout << "a = " << a->data << std::endl;

  std::cout << "\nGradients:" << std::endl;
  std::cout << "x1.grad = " << x1->grad << std::endl;
  std::cout << "x2.grad = " << x2->grad << std::endl;
  std::cout << "w1.grad = " << w1->grad << std::endl;
  std::cout << "w2.grad = " << w2->grad << std::endl;
  std::cout << "b.grad = " << b->grad << std::endl;

  // Verify results against known correct values
  assert(approx_equal(a->data, 0.7071));
  assert(approx_equal(w1->grad, 1.0));
  assert(approx_equal(w2->grad, 0.0));
  assert(approx_equal(b->grad, 0.5));

  auto n = std::make_shared<Value>(4.0);
  auto p = std::make_shared<Value>(2.0) * n->pow(std::make_shared<Value>(3.0));

  p->backprop();
  std::cout << "n.grad = " << n->grad << std::endl;
  assert(approx_equal(n->grad, 96.0));

  std::cout << "\nAll tests passed!" << std::endl;
}
