#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "lamppp/tensor/core.hpp"

using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Scalar;
using lmp::tensor::Tensor;

const Scalar kEps = 1e-5;

class TensorOpTest : public testing::Test {
 protected:
  TensorOpTest() = default;
  ~TensorOpTest() = default;

  void SetUp() override {
    tensor_f32_A = Tensor(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        std::vector<size_t>{3u, 2u}, DeviceType::CUDA, DataType::Float32);
    tensor_f32_B = Tensor(
        std::vector<float>{0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f},
        std::vector<size_t>{3u, 2u}, DeviceType::CUDA, DataType::Float32);
    std::vector<int32_t> data_i32_C_vec = {1, 2, 3, 4, 5, 6};
    tensor_i32_C = Tensor(data_i32_C_vec, std::vector<size_t>{3u, 2u},
                          DeviceType::CUDA, DataType::Int32);

    tensor_f32_1x2_broadcast =
        Tensor(std::vector<float>{10.0f, 20.0f}, std::vector<size_t>{1u, 2u},
               DeviceType::CUDA, DataType::Float32);
    tensor_f32_3x1_broadcast = Tensor(std::vector<float>{10.0f, 20.0f, 30.0f},
                                      std::vector<size_t>{3u, 1u},
                                      DeviceType::CUDA, DataType::Float32);
    scalar_tensor_f32 =
        Tensor(std::vector<Scalar>{100.0}, std::vector<size_t>{1},
               DeviceType::CUDA, DataType::Float32);
    tensor_f32_1x2x1_squeeze_expand =
        Tensor(std::vector<float>{7.0f, 8.0f}, std::vector<size_t>{1u, 2u, 1u},
               DeviceType::CUDA, DataType::Float32);
  }
  void TearDown() override {};
  std::vector<Scalar> getTenData(const Tensor& ten) {
    return ten.to_vector<Scalar>();
  }
  template <typename T>
  std::vector<T> getIntegerTenData(const Tensor& ten) {
    return ten.to_vector<T>();
  }

  Tensor tensor_f32_A;
  Tensor tensor_f32_B;
  Tensor tensor_i32_C;
  Tensor tensor_f32_1x2_broadcast;
  Tensor tensor_f32_3x1_broadcast;
  Tensor scalar_tensor_f32;
  Tensor tensor_f32_1x2x1_squeeze_expand;
};

TEST_F(TensorOpTest, TypeUpcastTest) {
  Tensor result = tensor_f32_A + tensor_i32_C;

  EXPECT_EQ(result.type(), DataType::Float32) << "Result data type mismatch";
  EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Result shape mismatch";

  std::vector<Scalar> expected_values = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
  EXPECT_THAT(getTenData(result),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Result data mismatch";

  Tensor result_rev = tensor_i32_C + tensor_f32_A;
  EXPECT_EQ(result_rev.type(), DataType::Float32)
      << "Reversed op: Result data type mismatch";
  EXPECT_THAT(result_rev.shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Reversed op: Result shape mismatch";
  EXPECT_THAT(getTenData(result_rev),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Reversed op: Result data mismatch";
}
TEST_F(TensorOpTest, SimpleBroadcastTest) {
  {
    Tensor result = tensor_f32_A + scalar_tensor_f32;
    EXPECT_EQ(result.type(), DataType::Float32)
        << "Scalar broadcast: Result data type mismatch";
    EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3u, 2u}))
        << "Scalar broadcast: Result shape mismatch";
    std::vector<Scalar> expected_values = {101.0f, 102.0f, 103.0f,
                                           104.0f, 105.0f, 106.0f};
    EXPECT_THAT(
        getTenData(result),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
        << "Scalar broadcast: Result data mismatch";
  }

  {
    Tensor result = tensor_f32_A + tensor_f32_1x2_broadcast;
    EXPECT_EQ(result.type(), DataType::Float32)
        << "Row broadcast: Result data type mismatch";
    EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3u, 2u}))
        << "Row broadcast: Result shape mismatch";
    std::vector<Scalar> expected_values = {11.0f, 22.0f, 13.0f,
                                           24.0f, 15.0f, 26.0f};
    EXPECT_THAT(
        getTenData(result),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
        << "Row broadcast: Result data mismatch";
  }

  {
    Tensor result = tensor_f32_A + tensor_f32_3x1_broadcast;
    EXPECT_EQ(result.type(), DataType::Float32)
        << "Column broadcast: Result data type mismatch";
    EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3u, 2u}))
        << "Column broadcast: Result shape mismatch";
    std::vector<Scalar> expected_values = {11.0f, 12.0f, 23.0f,
                                           24.0f, 35.0f, 36.0f};
    EXPECT_THAT(
        getTenData(result),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
        << "Column broadcast: Result data mismatch";
  }
}
TEST_F(TensorOpTest, ReshapeBroadcastTest) {
  Tensor flat_tensor =
      Tensor(std::vector<float>{10.f, 20.f, 30.f, 40.f, 50.f, 60.f},
             std::vector<size_t>{6u}, DeviceType::CUDA, DataType::Float32);
  Tensor reshaped_tensor = flat_tensor.reshape({3u, 2u});

  EXPECT_THAT(reshaped_tensor.shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Reshaped tensor shape mismatch";
  std::vector<Scalar> expected_reshaped_values = {10.f, 20.f, 30.f,
                                                  40.f, 50.f, 60.f};
  EXPECT_THAT(getTenData(reshaped_tensor),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   expected_reshaped_values))
      << "Reshaped tensor data mismatch";
  Tensor result = reshaped_tensor + tensor_f32_1x2_broadcast;

  EXPECT_EQ(result.type(), DataType::Float32)
      << "Reshape-Broadcast: Result data type mismatch";
  EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Reshape-Broadcast: Result shape mismatch";

  std::vector<Scalar> expected_final_values = {20.0f, 40.0f, 40.0f,
                                               60.0f, 60.0f, 80.0f};
  EXPECT_THAT(
      getTenData(result),
      ::testing::Pointwise(::testing::FloatNear(kEps), expected_final_values))
      << "Reshape-Broadcast: Result data mismatch";
}
TEST_F(TensorOpTest, ExpandBroadcastTest) {
  Tensor tensor_f32_A_expand = tensor_f32_A.expand_dims(2);
  Tensor result = tensor_f32_A_expand + tensor_f32_1x2x1_squeeze_expand;

  EXPECT_EQ(result.type(), DataType::Float32)
      << "Expand-Broadcast: Result data type mismatch";
  EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3u, 2u, 1u}))
      << "Expand-Broadcast: Result shape mismatch";

  std::vector<Scalar> expected_values = {1.0f + 7.0f, 2.0f + 8.0f,   // Row 0
                                         3.0f + 7.0f, 4.0f + 8.0f,   // Row 1
                                         5.0f + 7.0f, 6.0f + 8.0f};  // Row 2

  EXPECT_THAT(getTenData(result),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Expand-Broadcast: Result data mismatch";

  Tensor result_rev = tensor_f32_1x2x1_squeeze_expand + tensor_f32_A_expand;
  EXPECT_EQ(result_rev.type(), DataType::Float32)
      << "Reversed Expand-Broadcast: Result data type mismatch";
  EXPECT_THAT(result_rev.shape(), ::testing::ElementsAreArray({3u, 2u, 1u}))
      << "Reversed Expand-Broadcast: Result shape mismatch";
  EXPECT_THAT(getTenData(result_rev),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Reversed Expand-Broadcast: Result data mismatch";
}
TEST_F(TensorOpTest, ReductSqueezeTest) {
  {
    Tensor sum_axis0_keepdims = lmp::tensor::ops::sum(tensor_f32_A, 0);
    EXPECT_EQ(sum_axis0_keepdims.type(), DataType::Float32)
        << "Sum axis 0 (keepdims=true): Type mismatch";
    EXPECT_THAT(sum_axis0_keepdims.shape(),
                ::testing::ElementsAreArray({1u, 2u}))
        << "Sum axis 0 (keepdims=true): Shape mismatch";
    std::vector<Scalar> expected_sum_values_axis0 = {9.0f, 12.0f};
    EXPECT_THAT(getTenData(sum_axis0_keepdims),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     expected_sum_values_axis0))
        << "Sum axis 0 (keepdims=true): Data mismatch";

    Tensor sum_axis1_keepdims = lmp::tensor::ops::sum(tensor_f32_A, 1);
    EXPECT_EQ(sum_axis1_keepdims.type(), DataType::Float32)
        << "Sum axis 1 (keepdims=true): Type mismatch";
    EXPECT_THAT(sum_axis1_keepdims.shape(),
                ::testing::ElementsAreArray({3u, 1u}))
        << "Sum axis 1 (keepdims=true): Shape mismatch";
    std::vector<Scalar> expected_sum_values_axis1 = {3.0f, 7.0f, 11.0f};
    EXPECT_THAT(getTenData(sum_axis1_keepdims),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     expected_sum_values_axis1))
        << "Sum axis 1 (keepdims=true): Data mismatch";
  }

  {
    Tensor sum_axis0 = lmp::tensor::ops::sum(tensor_f32_A, 0);  // Shape {1,2}
    Tensor squeezed_sum_axis0 = sum_axis0.squeeze(0);
    EXPECT_EQ(squeezed_sum_axis0.type(), DataType::Float32)
        << "Sum axis 0 then squeeze: Type mismatch";
    EXPECT_THAT(squeezed_sum_axis0.shape(), ::testing::ElementsAreArray({2u}))
        << "Sum axis 0 then squeeze: Shape mismatch";
    std::vector<Scalar> expected_sum_values_axis0 = {9.0f, 12.0f};
    EXPECT_THAT(getTenData(squeezed_sum_axis0),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     expected_sum_values_axis0))
        << "Sum axis 0 then squeeze: Data mismatch";
  }
  {
    Tensor sum_axis1 = lmp::tensor::ops::sum(tensor_f32_A, 1);
    Tensor squeezed_sum_axis1 = sum_axis1.squeeze(1);
    EXPECT_EQ(squeezed_sum_axis1.type(), DataType::Float32)
        << "Sum axis 1 then squeeze: Type mismatch";
    EXPECT_THAT(squeezed_sum_axis1.shape(), ::testing::ElementsAreArray({3u}))
        << "Sum axis 1 then squeeze: Shape mismatch";
    std::vector<Scalar> expected_sum_values_axis1 = {3.0f, 7.0f, 11.0f};
    EXPECT_THAT(getTenData(squeezed_sum_axis1),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     expected_sum_values_axis1))
        << "Sum axis 1 then squeeze: Data mismatch";
  }

  {
    Tensor squeezed_ax0 = tensor_f32_1x2x1_squeeze_expand.squeeze(0);
    EXPECT_EQ(squeezed_ax0.type(), DataType::Float32)
        << "Squeeze ax0: Type mismatch";
    EXPECT_THAT(squeezed_ax0.shape(), ::testing::ElementsAreArray({2u, 1u}))
        << "Squeeze ax0: Shape mismatch";
    std::vector<Scalar> expected_squeeze_data = {7.0f, 8.0f};
    EXPECT_THAT(
        getTenData(squeezed_ax0),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_squeeze_data))
        << "Squeeze ax0: Data mismatch";

    Tensor squeezed_ax2 = tensor_f32_1x2x1_squeeze_expand.squeeze(2);
    EXPECT_EQ(squeezed_ax2.type(), DataType::Float32)
        << "Squeeze ax2: Type mismatch";
    EXPECT_THAT(squeezed_ax2.shape(), ::testing::ElementsAreArray({1u, 2u}))
        << "Squeeze ax2: Shape mismatch";
    EXPECT_THAT(
        getTenData(squeezed_ax2),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_squeeze_data))
        << "Squeeze ax2: Data mismatch";

    Tensor temp_squeeze = tensor_f32_1x2x1_squeeze_expand.squeeze(0);
    Tensor final_squeezed = temp_squeeze.squeeze(1);
    EXPECT_EQ(final_squeezed.type(), DataType::Float32)
        << "Sequential squeeze: Type mismatch";
    EXPECT_THAT(final_squeezed.shape(), ::testing::ElementsAreArray({2u}))
        << "Sequential squeeze: Shape mismatch";
    EXPECT_THAT(
        getTenData(final_squeezed),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_squeeze_data))
        << "Sequential squeeze: Data mismatch";
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
