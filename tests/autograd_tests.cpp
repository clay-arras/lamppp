#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "include/lamppp/autograd/core.hpp"
#include "include/lamppp/tensor/core.hpp"

using lmp::autograd::Variable;
using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Scalar;
using lmp::tensor::Tensor;

const Scalar kEps = 1e-5;

class VariableOpTest : public testing::Test {
 protected:
  VariableOpTest() = default;
  ~VariableOpTest() = default;

  void SetUp() override {
    a_data = Tensor(std::vector<Scalar>{1.0, 2.0, 3.0, 4.0, 5.0, 2.0},
                    std::vector<size_t>{3u, 2u}, DeviceType::CUDA,
                    DataType::Float32);
    b_data = Tensor(std::vector<Scalar>{-1.0, 4.0, -2.0, 0.0, 3.0, 0.5},
                    std::vector<size_t>{3u, 2u}, DeviceType::CUDA,
                    DataType::Float32);
    a = Variable(a_data, true);
    b = Variable(b_data, true);
  }
  void TearDown() override {};
  std::vector<Scalar> getTenData(Tensor ten) {
    std::span<Scalar> tmp = ten.view<Scalar>();
    return std::vector<Scalar>(tmp.begin(), tmp.end());
  }

  Variable a, b;
  Tensor a_data, b_data;
};

TEST_F(VariableOpTest, AddTest) {
  Variable res = a + b;
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {0.0, 6.0, 1.0, 4.0, 8.0, 2.5}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch for variable 1";
  EXPECT_THAT(getTenData(b.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch for variable 2";
}

TEST_F(VariableOpTest, SubTest) {
  Variable res = a - b;
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {2.0, -2.0, 5.0, 4.0, 2.0, 1.5}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch for variable 1";
  EXPECT_THAT(getTenData(b.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0}))
      << "Gradient mismatch for variable 2";
}

TEST_F(VariableOpTest, MulTest) {
  Variable res = a * b;
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0, 8.0, -6.0, 0.0, 15.0, 1.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0, 4.0, -2.0, 0.0, 3.0, 0.5}))
      << "Gradient mismatch for variable 1";
  EXPECT_THAT(getTenData(b.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 2.0, 3.0, 4.0, 5.0, 2.0}))
      << "Gradient mismatch for variable 2";
}

TEST_F(VariableOpTest, DivTest) {
  Variable res = a / b;
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(
                  ::testing::FloatNear(kEps),
                  {-1.0, 0.5, -1.5, std::numeric_limits<Scalar>::infinity(),
                   5.0 / 3.0, 4.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0 / -1.0, 1.0 / 4.0, 1.0 / -2.0,
                                    1.0 / 0.0, 1.0 / 3.0, 1.0 / 0.5}))
      << "Gradient mismatch for variable 1";
  EXPECT_THAT(getTenData(b.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0 / (-1.0 * -1.0), -2.0 / (4.0 * 4.0),
                                    -3.0 / (-2.0 * -2.0), -4.0 / (0.0 * 0.0),
                                    -5.0 / (3.0 * 3.0), -2.0 / (0.5 * 0.5)}))
      << "Gradient mismatch for variable 2";
}

TEST_F(VariableOpTest, ExpTest) {
  Variable res = lmp::autograd::ops::exp(b);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {exp(-1.0), exp(4.0), exp(-2.0), exp(0.0),
                                    exp(3.0), exp(0.5)}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(b.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {exp(-1.0), exp(4.0), exp(-2.0), exp(0.0),
                                    exp(3.0), exp(0.5)}))
      << "Gradient mismatch";
}

TEST_F(VariableOpTest, LogTest) {
  Variable res = lmp::autograd::ops::log(a);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(
                  ::testing::FloatNear(kEps),
                  {log(1.0), log(2.0), log(3.0), log(4.0), log(5.0), log(2.0)}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0 / 1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0,
                                    1.0 / 5.0, 1.0 / 2.0}))
      << "Gradient mismatch";
}

TEST_F(VariableOpTest, MatMulTest) {
  Tensor b_mat =
      Tensor(std::vector<Scalar>{-1.0, 4.0}, std::vector<size_t>{2u, 1u});
  Variable b_mat_var(b_mat, true);
  Variable res = lmp::autograd::ops::matmul(a, b_mat_var);
  EXPECT_THAT(
      getTenData(res.data()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {7.0, 13.0, 3.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u, 1u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0, 4.0, -1.0, 4.0, -1.0, 4.0}))
      << "Gradient mismatch for variable 1";
  EXPECT_THAT(getTenData(b_mat_var.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {9.0, 8.0}))
      << "Gradient mismatch for variable 2";
}

TEST_F(VariableOpTest, TransposeTest) {
  Variable res = lmp::autograd::ops::transpose(a);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 3.0, 5.0, 2.0, 4.0, 2.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({2u, 3u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch";
}

TEST_F(VariableOpTest, SumTest) {
  Variable res = lmp::autograd::ops::sum(a, 1);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {3.0, 7.0, 7.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u, 1u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
