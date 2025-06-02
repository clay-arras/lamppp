#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "lamppp/autograd/core.hpp"
#include "lamppp/autograd/functions/view_ops.hpp"
#include "lamppp/tensor/core.hpp"
#include "lamppp/tensor/device_type.hpp"

using lmp::autograd::Variable;
using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Scalar;
using lmp::tensor::Tensor;

const Scalar kEps = 1e-5;

class VariableOpTest
    : public testing::Test,
      public testing::WithParamInterface<std::tuple<DeviceType>> {
 protected:
  VariableOpTest() = default;
  ~VariableOpTest() = default;

  void SetUp() override {
    device = std::get<0>(GetParam());

    a_data = Tensor(std::vector<Scalar>{1.0, 2.0, 3.0, 4.0, 5.0, 2.0},
                    std::vector<size_t>{3u, 2u}, device, DataType::Float32);
    b_data = Tensor(std::vector<Scalar>{-1.0, 4.0, -2.0, 0.0, 3.0, 0.5},
                    std::vector<size_t>{3u, 2u}, device, DataType::Float32);
    a = Variable(a_data, true);
    b = Variable(b_data, true);
  }
  void TearDown() override {};
  std::vector<Scalar> getTenData(Tensor ten) { return ten.to_vector<Scalar>(); }

  Variable a, b;
  Tensor a_data, b_data;
  DeviceType device;
};

TEST_P(VariableOpTest, AddTest) {
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

TEST_P(VariableOpTest, SubTest) {
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

TEST_P(VariableOpTest, MulTest) {
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

TEST_P(VariableOpTest, DivTest) {
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

TEST_P(VariableOpTest, ExpTest) {
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

TEST_P(VariableOpTest, LogTest) {
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

TEST_P(VariableOpTest, MatMulTest) {
  Tensor b_mat = Tensor(std::vector<Scalar>{-1.0, 4.0},
                        std::vector<size_t>{2u, 1u}, device);
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

TEST_P(VariableOpTest, TransposeTest) {
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

TEST_P(VariableOpTest, SumTest) {
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

TEST_P(VariableOpTest, MaxTest) {
  Variable res = lmp::autograd::ops::max(a, 1);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {2.0, 4.0, 5.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u, 1u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(
      getTenData(a.grad()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {0, 1, 0, 1, 1, 0}))
      << "Gradient mismatch";
}

TEST_P(VariableOpTest, MinTest) {
  Variable res = lmp::autograd::ops::min(a, 1);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {1.0, 3.0, 2.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u, 1u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(
      getTenData(a.grad()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {1, 0, 1, 0, 0, 1}))
      << "Gradient mismatch";
}

TEST_P(VariableOpTest, ReshapeTest) {
  Variable res = lmp::autograd::ops::reshape(a, {2, 1, 3});
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 2.0, 3.0, 4.0, 5.0, 2.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({2u, 1u, 3u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(
      getTenData(a.grad()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {1, 1, 1, 1, 1, 1}))
      << "Gradient data mismatch";
  EXPECT_THAT(a.grad().shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Gradient shape mismatch";
}

TEST_P(VariableOpTest, ExpandDimsTest) {
  Variable res = lmp::autograd::ops::expand_dims(a, 0);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 2.0, 3.0, 4.0, 5.0, 2.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({1u, 3u, 2u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(
      getTenData(a.grad()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {1, 1, 1, 1, 1, 1}))
      << "Gradient data mismatch";
  EXPECT_THAT(a.grad().shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Gradient shape mismatch";
}

TEST_P(VariableOpTest, SqueezeTest) {
  Tensor squeeze_data =
      Tensor(std::vector<Scalar>{1.0, 2.0, 3.0}, std::vector<size_t>{3u, 1u},
             device, DataType::Float32);
  Variable squeeze_var = Variable(squeeze_data, true);
  Variable res = lmp::autograd::ops::squeeze(squeeze_var, 1);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {1.0, 2.0, 3.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3u}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(squeeze_var.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {1, 1, 1}))
      << "Gradient data mismatch";
  EXPECT_THAT(squeeze_var.grad().shape(), ::testing::ElementsAreArray({3u, 1u}))
      << "Gradient shape mismatch";
}

INSTANTIATE_TEST_SUITE_P(VariableOp, VariableOpTest,
                         testing::Values(DeviceType::CPU, DeviceType::CUDA));

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
