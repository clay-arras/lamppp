#include "../autograd/engine.h"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

// Helper function to check if two doubles are approximately equal
bool approx_equal(double a, double b, double epsilon = 1e-10) {
    return std::abs(a - b) < epsilon;
}

int main() {
    // Create input values
    auto x1 = std::make_shared<Value>(2.0);
    auto x2 = std::make_shared<Value>(-3.0);
    auto w1 = std::make_shared<Value>(-3.0);
    auto w2 = std::make_shared<Value>(1.0);
    auto b = std::make_shared<Value>(6.8813735870195432);
    auto a = std::make_shared<Value>(1.0);

    // Compute: z = w1*x1 + w2*x2 + b
    // Then: out = tanh(z)
    // Finally: L = (a - out)^2

    // Forward pass
    auto w1x1 = w1->operator*(x1);
    auto w2x2 = w2->operator*(x2);
    auto w1x1w2x2 = w1x1->operator+(w2x2);
    auto z = w1x1w2x2->operator+(b);
    auto out = z->tanh();
    auto diff = a->operator-(out);
    auto L = diff->pow(2);

    // Backward pass
    L->backprop();

    // Print results
    std::cout << std::fixed << std::setprecision(10);
    
    std::cout << "Forward pass values:" << std::endl;
    std::cout << "w1*x1 = " << w1x1->data << std::endl;
    std::cout << "w2*x2 = " << w2x2->data << std::endl;
    std::cout << "z = " << z->data << std::endl;
    std::cout << "out = " << out->data << std::endl;
    std::cout << "L = " << L->data << std::endl;

    std::cout << "\nGradients:" << std::endl;
    std::cout << "x1.grad = " << x1->grad << std::endl;
    std::cout << "x2.grad = " << x2->grad << std::endl;
    std::cout << "w1.grad = " << w1->grad << std::endl;
    std::cout << "w2.grad = " << w2->grad << std::endl;
    std::cout << "b.grad = " << b->grad << std::endl;

    // Verify results against known correct values
    assert(approx_equal(L->data, 0.5));
    assert(approx_equal(x1->grad, 1.0));
    assert(approx_equal(w1->grad, 2.0));
    assert(approx_equal(b->grad, 0.5));

    std::cout << "\nAll tests passed!" << std::endl;

    return 0;
}
