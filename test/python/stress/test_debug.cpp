#include <cassert>
#include <iostream>
#include <vector>
#include <cmath> // For std::exp, std::log

#include "autograd/engine/tensor.h"
#include "autograd/engine/variable.h"

using autograd::Tensor;
using autograd::Variable;

// Helper function to create tensors more easily
Tensor make_tensor(const std::vector<float>& data, const std::vector<int>& shape) {
    // Assuming Tensor has a constructor like this.
    // If not, this needs to be adjusted based on tensor.h API.
    return Tensor(data, shape);
}


void test_add() {
    std::cout << "--- Testing Add ---" << std::endl;
    Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    Tensor t2 = make_tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
    Variable v1(t1, true);
    Variable v2(t2, true);

    Variable add_res = v1 + v2;
    add_res.backward();

    assert(add_res.data().shape() == std::vector<int>({3, 2}));
    assert(!v1.grad().shape().empty()); // Check if grad tensor shape is not empty
    assert(!v2.grad().shape().empty());
    std::cout << "Add test passed." << std::endl;
}

void test_sub() {
    std::cout << "--- Testing Sub ---" << std::endl;
    Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    Tensor t2 = make_tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
    Variable v1(t1, true);
    Variable v2(t2, true);

    Variable sub_res = v1 - v2;
    sub_res.backward();

    assert(sub_res.data().shape() == std::vector<int>({3, 2}));
    assert(!v1.grad().shape().empty());
    assert(!v2.grad().shape().empty());
    std::cout << "Sub test passed." << std::endl;
}

void test_mul() {
    std::cout << "--- Testing Mul ---" << std::endl;
    Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    Tensor t2 = make_tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
    Variable v1(t1, true);
    Variable v2(t2, true);

    Variable mul_res = v1 * v2; // Element-wise
    mul_res.backward();

    assert(mul_res.data().shape() == std::vector<int>({3, 2}));
    assert(!v1.grad().shape().empty());
    assert(!v2.grad().shape().empty());
    std::cout << "Mul test passed." << std::endl;
}

void test_div() {
    std::cout << "--- Testing Div ---" << std::endl;
    Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    // Ensure non-zero divisor
    Tensor t2 = make_tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
    Variable v1(t1, true);
    Variable v2(t2, true);

    Variable div_res = v1 / v2;
    div_res.backward();

    assert(div_res.data().shape() == std::vector<int>({3, 2}));
    assert(!v1.grad().shape().empty());
    assert(!v2.grad().shape().empty());
    std::cout << "Div test passed." << std::endl;
}

void test_relu() {
    std::cout << "--- Testing ReLU ---" << std::endl;
    Tensor t1 = make_tensor({-1.0f, 2.0f, -3.0f, 4.0f, 0.0f, -6.0f}, {3, 2});
    Variable v1(t1, true);

    Variable relu_res = v1.relu();
    relu_res.backward();

    assert(relu_res.data().shape() == std::vector<int>({3, 2}));
    assert(!v1.grad().shape().empty());
    std::cout << "ReLU test passed." << std::endl;
}

void test_exp() {
     std::cout << "--- Testing Exp ---" << std::endl;
    Tensor t1 = make_tensor({1.0f, 2.0f, 0.0f, -1.0f, 5.0f, -2.0f}, {3, 2});
    Variable v1(t1, true);

    Variable exp_res = v1.exp();
    exp_res.backward();

    assert(exp_res.data().shape() == std::vector<int>({3, 2}));
    assert(!v1.grad().shape().empty());
    std::cout << "Exp test passed." << std::endl;
}

void test_log() {
    std::cout << "--- Testing Log ---" << std::endl;
    // Use positive values for log
    Tensor t_pos = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    Variable v_pos(t_pos, true);

    Variable log_res = v_pos.log();
    log_res.backward();

    assert(log_res.data().shape() == std::vector<int>({3, 2}));
    assert(!v_pos.grad().shape().empty());
    std::cout << "Log test passed." << std::endl;
}


void test_matmul() {
    std::cout << "--- Testing MatMul ---" << std::endl;
    Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}); // 3x2
    Tensor t3 = make_tensor({1.0f, 2.0f}, {2, 1}); // 2x1
    Variable v1(t1, true);
    Variable v3(t3, true);

    // v1 (3x2) @ v3 (2x1) -> 3x1
    Variable matmul_res = v1.matmul(v3);
    matmul_res.backward(); // Requires reduction or further op for backward usually, assume sum reduction internally for backward

    assert(matmul_res.data().shape() == std::vector<int>({3, 1}));
    assert(!v1.grad().shape().empty());
    assert(!v3.grad().shape().empty());
    std::cout << "MatMul test passed." << std::endl;
}

void test_transpose() {
     std::cout << "--- Testing Transpose ---" << std::endl;
    Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}); // 3x2
    Variable v1(t1, true);

    // v1 (3x2) -> 2x3
    Variable transpose_res = v1.transpose();

    // Backward on transpose typically needs a reduction first to become scalar.
    // We'll sum the result twice to enable backward().
    Variable scalar_output = transpose_res.sum(0).sum(0); // Sum along both axes
    scalar_output.backward();

    assert(transpose_res.data().shape() == std::vector<int>({2, 3}));
    assert(!v1.grad().shape().empty()); // Grad should be defined now
    std::cout << "Transpose test passed." << std::endl;
}


void test_sum() {
    std::cout << "--- Testing Sum ---" << std::endl;
    Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}); // 3x2
    Variable v1(t1, true);

    // Sum v1 (3x2) along axis 1 -> {3}
    Variable sum_res_axis1 = v1.sum(1);
    // To backprop through sum, need to sum the result again to get scalar
    Variable scalar_sum1 = sum_res_axis1.sum(0); // Sum the resulting {3} vector
    scalar_sum1.backward();
    // Check shape might depend on implementation (keepdims=True/False)
    // Assuming it reduces the dimension
    assert(sum_res_axis1.data().shape() == std::vector<int>({3}));
    assert(!v1.grad().shape().empty());
    std::cout << "Sum (axis=1) test passed." << std::endl;
    v1.zero_grad(); // Reset grad for next sum test

    // Sum v1 (3x2) along axis 0 -> {2} (assuming shape becomes {2})
    Variable sum_res_axis0 = v1.sum(0);
    Variable scalar_sum0 = sum_res_axis0.sum(0); // Sum the resulting {2} vector
    scalar_sum0.backward();
    // Assuming it reduces the dimension
    assert(sum_res_axis0.data().shape() == std::vector<int>({2}));
    assert(!v1.grad().shape().empty());
    std::cout << "Sum (axis=0) test passed." << std::endl;
    v1.zero_grad();

    // Sum all elements by summing axis 0, then axis 0 of the result
     Variable sum_res_all = v1.sum(0).sum(0); // Sum axis 0 ({1,2}), then sum axis 0 ({1})
     sum_res_all.backward();
     // Check if the final result is scalar (shape {}) or {1}
     assert(sum_res_all.data().shape().empty() || sum_res_all.data().shape() == std::vector<int>({1}));
     assert(!v1.grad().shape().empty());
     std::cout << "Sum (all) test passed." << std::endl;
}


int main() {
    test_add();
    test_sub();
    test_mul();
    test_div();
    test_relu();
    test_exp();
    test_log();
    test_matmul();
    test_transpose();
    test_sum();

    std::cout << "\nAll tests finished." << std::endl;

    return 0;
}
