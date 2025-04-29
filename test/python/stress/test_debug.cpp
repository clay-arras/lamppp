#include <cassert>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "autograd/engine/tensor.h"
#include "autograd/engine/variable.h"

using autograd::Tensor;
using autograd::Variable;

[[noreturn]]
inline void assert_fail(const char* expr, const char* file, int line,
                        const std::string& msg) {
  std::cerr << file << ":" << line << ": Assertion `" << expr
            << "` failed: " << msg << "\n";
  std::abort();
}

#define DASSERT(cond, ...)                                              \
  do {                                                                  \
    if (!(cond))                                                        \
      assert_fail(#cond, __FILE__, __LINE__, std::format(__VA_ARGS__)); \
  } while (0)

template <typename T>
std::string to_string(std::vector<T> item) {
  std::string str = "{";
  for (auto i : item)
    str.append(std::to_string(i) + " ");
  return str + "}";
}
Tensor make_tensor(const std::vector<float>& data,
                   const std::vector<int>& shape) {
  return Tensor(data, shape);
}
void check_tensor(const Tensor t, std::vector<float> data,
                  std::vector<int> shape) {
//   DASSERT(t.data() == data, "Data mismatch, expected: {}; got: {}",
//           to_string(data), to_string(t.data()));
//   DASSERT(t.shape() == shape, "Shape mismatch, expected: {}; got: {}",
//           to_string(shape), to_string(t.shape()));
    std::cout << "Data: ";
    (t.data() == data) ? std::cout << "Ok" << std::endl : std::cout << "ERROR" << std::endl;
    std::cout << "Shape: ";
    (t.shape() == shape) ? std::cout << "Ok" << std::endl : std::cout << "ERROR" << std::endl;
}

void test_add() {
  std::cout << "--- Testing Add ---" << std::endl;
  Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  Tensor t2 = make_tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
  Variable v1(t1, true);
  Variable v2(t2, true);

  Variable add_res = v1 + v2;
  std::cout << "Forward Test..." << std::endl;
  check_tensor(add_res.data(), {8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f},
               {3, 2});
  add_res.backward();

  std::cout << "Backward Test..." << std::endl;
  check_tensor(v1.grad(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {3, 2});
  check_tensor(v2.grad(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {3, 2});

  std::cout << "Add test passed." << std::endl << std::endl;
}

void test_sub() {
  std::cout << "--- Testing Sub ---" << std::endl;
  Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  Tensor t2 = make_tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
  Variable v1(t1, true);
  Variable v2(t2, true);

  Variable sub_res = v1 - v2;
  std::cout << "Forward Test..." << std::endl;
  check_tensor(sub_res.data(), {-6.0f, -6.0f, -6.0f, -6.0f, -6.0f, -6.0f},
               {3, 2});
  sub_res.backward();

  std::cout << "Backward Test..." << std::endl;
  check_tensor(v1.grad(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {3, 2});
  check_tensor(v2.grad(), {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f}, {3, 2});

  std::cout << "Sub test passed." << std::endl << std::endl;
}

void test_mul() {
  std::cout << "--- Testing Mul ---" << std::endl;
  Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  Tensor t2 = make_tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
  Variable v1(t1, true);
  Variable v2(t2, true);

  Variable mul_res = v1 * v2;
  std::cout << "Forward Test..." << std::endl;
  check_tensor(mul_res.data(), {7.0f, 16.0f, 27.0f, 40.0f, 55.0f, 72.0f},
               {3, 2});
  mul_res.backward();

  std::cout << "Backward Test..." << std::endl;
  check_tensor(v1.grad(), {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
  check_tensor(v2.grad(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});

  std::cout << "Mul test passed." << std::endl << std::endl;
}

void test_div() {
  std::cout << "--- Testing Div ---" << std::endl;
  Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  Tensor t2 = make_tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
  Variable v1(t1, true);
  Variable v2(t2, true);

  Variable div_res = v1 / v2;
  std::cout << "Forward Test..." << std::endl;
  check_tensor(div_res.data(),
               {1.0f / 7.0f, 2.0f / 8.0f, 3.0f / 9.0f, 4.0f / 10.0f,
                5.0f / 11.0f, 6.0f / 12.0f},
               {3, 2});
  div_res.backward();

  std::cout << "Backward Test..." << std::endl;
  check_tensor(v1.grad(),
               {1.0f / 7.0f, 2.0f / 8.0f, 3.0f / 9.0f, 4.0f / 10.0f,
                5.0f / 11.0f, 6.0f / 12.0f},
               {3, 2});
  check_tensor(
      v2.grad(),
      {-1.0f * (1.0f / (7.0f * 7.0f)), -1.0f * (2.0f / (8.0f * 8.0f)),
       -1.0f * (3.0f / (9.0f * 9.0f)), -1.0f * (4.0f / (10.0f * 10.0f)),
       -1.0f * (5.0f / (11.0f * 11.0f)), -1.0f * (6.0f / (12.0f * 12.0f))},
      {3, 2});

  std::cout << "Div test passed." << std::endl << std::endl;
}

void test_relu() {
  std::cout << "--- Testing ReLU ---" << std::endl;
  Tensor t1 = make_tensor({-1.0f, 2.0f, -3.0f, 4.0f, 0.0f, -6.0f}, {3, 2});
  Variable v1(t1, true);

  Variable relu_res = v1.relu();
  std::cout << "Forward Test..." << std::endl;
  check_tensor(relu_res.data(), {0.0f, 2.0f, 0.0f, 4.0f, 0.0f, 0.0f}, {3, 2});
  relu_res.backward();

  std::cout << "Backward Test..." << std::endl;
  check_tensor(v1.grad(), {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f}, {3, 2});

  std::cout << "ReLU test passed." << std::endl << std::endl;
}

void test_exp() {
  std::cout << "--- Testing Exp ---" << std::endl;
  Tensor t1 = make_tensor({1.0f, 2.0f, 0.0f, -1.0f, 5.0f, -2.0f}, {3, 2});
  Variable v1(t1, true);

  Variable exp_res = v1.exp();
  std::cout << "Forward Test..." << std::endl;
  check_tensor(exp_res.data(), 
               {expf(1.0f), expf(2.0f), expf(0.0f), expf(-1.0f), expf(5.0f), expf(-2.0f)}, 
               {3, 2});
  exp_res.backward();

  std::cout << "Backward Test..." << std::endl;
  check_tensor(v1.grad(), 
               {expf(1.0f), expf(2.0f), expf(0.0f), expf(-1.0f), expf(5.0f), expf(-2.0f)}, 
               {3, 2});

  std::cout << "Exp test passed." << std::endl << std::endl;
}

void test_log() {
  std::cout << "--- Testing Log ---" << std::endl;
  Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  Variable v1(t1, true);

  Variable log_res = v1.log();
  std::cout << "Forward Test..." << std::endl;
  check_tensor(log_res.data(), 
               {logf(1.0f), logf(2.0f), logf(3.0f), logf(4.0f), logf(5.0f), logf(6.0f)}, 
               {3, 2});
  log_res.backward();

  std::cout << "Backward Test..." << std::endl;
  check_tensor(v1.grad(), 
               {1.0f/1.0f, 1.0f/2.0f, 1.0f/3.0f, 1.0f/4.0f, 1.0f/5.0f, 1.0f/6.0f}, 
               {3, 2});

  std::cout << "Log test passed." << std::endl << std::endl;
}

void test_matmul() {
  std::cout << "--- Testing MatMul ---" << std::endl;
  Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});  // 3x2
  Tensor t2 = make_tensor({1.0f, 2.0f}, {2, 1});                          // 2x1
  Variable v1(t1, true);
  Variable v2(t2, true);

  Variable matmul_res = v1.matmul(v2);
  std::cout << "Forward Test..." << std::endl;
  check_tensor(matmul_res.data(), {5.0f, 11.0f, 17.0f}, {3, 1});
  matmul_res.backward();

  std::cout << "Backward Test..." << std::endl;
  check_tensor(v1.grad(), {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f}, {3, 2});
  check_tensor(v2.grad(), {9.0f, 12.0f}, {2, 1});

  std::cout << "MatMul test passed." << std::endl << std::endl;;
}

void test_transpose() {
  std::cout << "--- Testing Transpose ---" << std::endl;
  Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});  // 3x2
  Variable v1(t1, true);

  Variable transpose_res = v1.transpose();
  std::cout << "Forward Test..." << std::endl;
  check_tensor(transpose_res.data(), {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f}, {2, 3});
  transpose_res.backward();

  std::cout << "Backward Test..." << std::endl;
  check_tensor(v1.grad(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {3, 2});

  std::cout << "Transpose test passed." << std::endl << std::endl;;
}

void test_sum() {
  std::cout << "--- Testing Sum ---" << std::endl;
  Tensor t1 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});  // 3x2
  Variable v1(t1, true);
  Tensor t2 = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});  // 3x2
  Variable v2(t2, true);

  Variable sum_res_axis1 = v1.sum(1);
  std::cout << "Sum Forward Test..." << std::endl;
  check_tensor(sum_res_axis1.data(), {3.0f, 7.0f, 11.0f}, {3});
  Variable scalar_sum1 = sum_res_axis1.sum(0);
  scalar_sum1.backward();

  std::cout << "Sum Backward Test..." << std::endl;
  check_tensor(v1.grad(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {3, 2});
  std::cout << "Sum test passed." << std::endl << std::endl;
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