#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "lamppp/tensor/core.hpp"
#include "lamppp/tensor/data_type.hpp"

using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Scalar;
using lmp::tensor::Tensor;

const Scalar kEps = 1e-5;

using ParamTypes = std::tuple<DeviceType, DataType>;

class TensorOpTest : public testing::Test,
                     public testing::WithParamInterface<ParamTypes> {
 protected:
  TensorOpTest() = default;
  ~TensorOpTest() = default;

  void SetUp() override {
    device = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());

    tensor_f32_A =
        Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
               std::vector<size_t>{3u, 2u}, device, dtype);
    tensor_f32_B =
        Tensor(std::vector<float>{0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f},
               std::vector<size_t>{3u, 2u}, device, dtype);

    tensor_f32_1x2_broadcast =
        Tensor(std::vector<float>{10.0f, 20.0f}, std::vector<size_t>{1u, 2u},
               device, dtype);
    tensor_f32_3x1_broadcast =
        Tensor(std::vector<float>{10.0f, 20.0f, 30.0f},
               std::vector<size_t>{3u, 1u}, device, dtype);
    scalar_tensor_f32 = Tensor(std::vector<Scalar>{100.0},
                               std::vector<size_t>{1}, device, dtype);
    tensor_f32_1x2x1_squeeze_expand =
        Tensor(std::vector<float>{7.0f, 8.0f}, std::vector<size_t>{1u, 2u, 1u},
               device, dtype);
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
  Tensor tensor_f32_1x2_broadcast;
  Tensor tensor_f32_3x1_broadcast;
  Tensor scalar_tensor_f32;
  Tensor tensor_f32_1x2x1_squeeze_expand;

  DeviceType device;
  DataType dtype;
};

TEST_P(TensorOpTest, TypeUpcastTest) {
  Tensor tensor_i32_C;
  std::vector<int32_t> data_i32_C_vec = {1, 2, 3, 4, 5, 6};
  tensor_i32_C = Tensor(data_i32_C_vec, std::vector<size_t>{3u, 2u}, device,
                        DataType::Int32);
  Tensor result = tensor_f32_A + tensor_i32_C;

  EXPECT_EQ(result.type(), lmp::tensor::type_upcast(tensor_i32_C.type(), dtype))
      << "Result data type mismatch";
  EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Result shape mismatch";

  std::vector<Scalar> expected_values = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
  EXPECT_THAT(getTenData(result),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Result data mismatch";

  Tensor result_rev = tensor_i32_C + tensor_f32_A;
  EXPECT_EQ(result_rev.type(),
            lmp::tensor::type_upcast(tensor_i32_C.type(), dtype))
      << "Reversed op: Result data type mismatch";
  EXPECT_THAT(result_rev.shape(), ::testing::ElementsAreArray({3u, 2u}))
      << "Reversed op: Result shape mismatch";
  EXPECT_THAT(getTenData(result_rev),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Reversed op: Result data mismatch";
}
TEST_P(TensorOpTest, SimpleBroadcastTest) {
  {
    Tensor result = tensor_f32_A + scalar_tensor_f32;
    EXPECT_EQ(result.type(), dtype)
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
    EXPECT_EQ(result.type(), dtype)
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
    EXPECT_EQ(result.type(), dtype)
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
TEST_P(TensorOpTest, ReshapeBroadcastTest) {
  Tensor flat_tensor =
      Tensor(std::vector<float>{10.f, 20.f, 30.f, 40.f, 50.f, 60.f},
             std::vector<size_t>{6u}, device, dtype);
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

  EXPECT_EQ(result.type(), dtype)
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
TEST_P(TensorOpTest, ExpandBroadcastTest) {
  Tensor tensor_f32_A_expand = tensor_f32_A.expand_dims(2);
  Tensor result = tensor_f32_A_expand + tensor_f32_1x2x1_squeeze_expand;

  EXPECT_EQ(result.type(), dtype)
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
  EXPECT_EQ(result_rev.type(), dtype)
      << "Reversed Expand-Broadcast: Result data type mismatch";
  EXPECT_THAT(result_rev.shape(), ::testing::ElementsAreArray({3u, 2u, 1u}))
      << "Reversed Expand-Broadcast: Result shape mismatch";
  EXPECT_THAT(getTenData(result_rev),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Reversed Expand-Broadcast: Result data mismatch";
}
TEST_P(TensorOpTest, ReductSqueezeTest) {
  {
    Tensor sum_axis0_keepdims = lmp::tensor::ops::sum(tensor_f32_A, 0);
    EXPECT_EQ(sum_axis0_keepdims.type(), dtype)
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
    EXPECT_EQ(sum_axis1_keepdims.type(), dtype)
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
    EXPECT_EQ(squeezed_sum_axis0.type(), dtype)
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
    EXPECT_EQ(squeezed_sum_axis1.type(), dtype)
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
    EXPECT_EQ(squeezed_ax0.type(), dtype) << "Squeeze ax0: Type mismatch";
    EXPECT_THAT(squeezed_ax0.shape(), ::testing::ElementsAreArray({2u, 1u}))
        << "Squeeze ax0: Shape mismatch";
    std::vector<Scalar> expected_squeeze_data = {7.0f, 8.0f};
    EXPECT_THAT(
        getTenData(squeezed_ax0),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_squeeze_data))
        << "Squeeze ax0: Data mismatch";

    Tensor squeezed_ax2 = tensor_f32_1x2x1_squeeze_expand.squeeze(2);
    EXPECT_EQ(squeezed_ax2.type(), dtype) << "Squeeze ax2: Type mismatch";
    EXPECT_THAT(squeezed_ax2.shape(), ::testing::ElementsAreArray({1u, 2u}))
        << "Squeeze ax2: Shape mismatch";
    EXPECT_THAT(
        getTenData(squeezed_ax2),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_squeeze_data))
        << "Squeeze ax2: Data mismatch";

    Tensor temp_squeeze = tensor_f32_1x2x1_squeeze_expand.squeeze(0);
    Tensor final_squeezed = temp_squeeze.squeeze(1);
    EXPECT_EQ(final_squeezed.type(), dtype)
        << "Sequential squeeze: Type mismatch";
    EXPECT_THAT(final_squeezed.shape(), ::testing::ElementsAreArray({2u}))
        << "Sequential squeeze: Shape mismatch";
    EXPECT_THAT(
        getTenData(final_squeezed),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_squeeze_data))
        << "Sequential squeeze: Data mismatch";
  }
}
TEST_P(TensorOpTest, ToTest) {
  if (device == DeviceType::CPU) {
    Tensor result = tensor_f32_B.to(DeviceType::CUDA);

    EXPECT_EQ(result.device(), DeviceType::CUDA)
        << "To: Result device mismatch";
    EXPECT_NE(result.device(), tensor_f32_B.device())
        << "To: Result device mismatch";
    EXPECT_THAT(getTenData(result),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_B)))
        << "To: Result data mismatch";
  } else if (device == DeviceType::CUDA) {
    Tensor result = tensor_f32_B.to(DeviceType::CPU);

    EXPECT_EQ(result.device(), DeviceType::CPU) << "To: Result device mismatch";
    EXPECT_NE(result.device(), tensor_f32_B.device())
        << "To: Result device mismatch";
    EXPECT_THAT(getTenData(result),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_B)))
        << "To: Result data mismatch";
  } else {
    ASSERT_TRUE(false);
  }
}
TEST_P(TensorOpTest, CopyTest) {
  {
    Tensor tensor_copy_target =
        Tensor(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
               std::vector<size_t>{3u, 2u}, device, dtype);

    tensor_copy_target.copy(tensor_f32_A);

    EXPECT_EQ(tensor_copy_target.type(), dtype)
        << "Copy: Result data type mismatch";
    EXPECT_THAT(tensor_copy_target.shape(),
                ::testing::ElementsAreArray({3u, 2u}))
        << "Copy: Result shape mismatch";
    EXPECT_THAT(getTenData(tensor_copy_target),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_A)))
        << "Copy: Result data mismatch";
  }

  {
    std::vector<int32_t> data_i32_target = {0, 0, 0, 0, 0, 0};
    Tensor tensor_i32_target = Tensor(
        data_i32_target, std::vector<size_t>{3u, 2u}, device, DataType::Int32);

    tensor_i32_target.copy(tensor_f32_A);

    EXPECT_EQ(tensor_i32_target.type(), DataType::Int32)
        << "Copy with type conversion: Result data type mismatch";
    EXPECT_THAT(tensor_i32_target.shape(),
                ::testing::ElementsAreArray({3u, 2u}))
        << "Copy with type conversion: Result shape mismatch";

    std::vector<int32_t> expected_values = {1, 2, 3, 4, 5, 6};
    EXPECT_THAT(getIntegerTenData<int32_t>(tensor_i32_target),
                ::testing::ElementsAreArray(expected_values))
        << "Copy with type conversion: Result data mismatch";
  }

  if (device == DeviceType::CPU) {
    Tensor tensor_cuda_target =
        Tensor(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
               std::vector<size_t>{3u, 2u}, DeviceType::CUDA, dtype);

    tensor_cuda_target.copy(tensor_f32_A);

    EXPECT_EQ(tensor_cuda_target.device(), DeviceType::CUDA)
        << "Copy cross-device: Result device mismatch";
    EXPECT_THAT(getTenData(tensor_cuda_target),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_A)))
        << "Copy cross-device: Result data mismatch";
  } else if (device == DeviceType::CUDA) {
    Tensor tensor_cpu_target =
        Tensor(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
               std::vector<size_t>{3u, 2u}, DeviceType::CPU, dtype);

    tensor_cpu_target.copy(tensor_f32_A);

    EXPECT_EQ(tensor_cpu_target.device(), DeviceType::CPU)
        << "Copy cross-device: Result device mismatch";
    EXPECT_THAT(getTenData(tensor_cpu_target),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_A)))
        << "Copy cross-device: Result data mismatch";
  }
}
TEST_P(TensorOpTest, IndexTest) {
  {
    Scalar indexed_value = tensor_f32_A.index({0, 0});
    EXPECT_NEAR(indexed_value, 1.0f, kEps) << "Index [0,0]: Value mismatch";

    indexed_value = tensor_f32_A.index({0, 1});
    EXPECT_NEAR(indexed_value, 2.0f, kEps) << "Index [0,1]: Value mismatch";

    indexed_value = tensor_f32_A.index({1, 0});
    EXPECT_NEAR(indexed_value, 3.0f, kEps) << "Index [1,0]: Value mismatch";

    indexed_value = tensor_f32_A.index({2, 1});
    EXPECT_NEAR(indexed_value, 6.0f, kEps) << "Index [2,1]: Value mismatch";
  }

  {
    Scalar scalar_value = scalar_tensor_f32.index({0});
    EXPECT_NEAR(scalar_value, 100.0f, kEps)
        << "Index scalar tensor: Value mismatch";
  }

  {
    Scalar indexed_3d_value = tensor_f32_1x2x1_squeeze_expand.index({0, 0, 0});
    EXPECT_NEAR(indexed_3d_value, 7.0f, kEps)
        << "Index 3D tensor [0,0,0]: Value mismatch";

    indexed_3d_value = tensor_f32_1x2x1_squeeze_expand.index({0, 1, 0});
    EXPECT_NEAR(indexed_3d_value, 8.0f, kEps)
        << "Index 3D tensor [0,1,0]: Value mismatch";
  }

  {
    Scalar broadcast_1x2_val = tensor_f32_1x2_broadcast.index({0, 0});
    EXPECT_NEAR(broadcast_1x2_val, 10.0f, kEps)
        << "Index 1x2 broadcast [0,0]: Value mismatch";

    broadcast_1x2_val = tensor_f32_1x2_broadcast.index({0, 1});
    EXPECT_NEAR(broadcast_1x2_val, 20.0f, kEps)
        << "Index 1x2 broadcast [0,1]: Value mismatch";

    Scalar broadcast_3x1_val = tensor_f32_3x1_broadcast.index({1, 0});
    EXPECT_NEAR(broadcast_3x1_val, 20.0f, kEps)
        << "Index 3x1 broadcast [1,0]: Value mismatch";
  }
}
TEST_P(TensorOpTest, FillTest) {
  {
    Tensor tensor_to_fill =
        Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
               std::vector<size_t>{3u, 2u}, device, dtype);

    Scalar fill_value = 42.0;
    tensor_to_fill.fill(fill_value);

    EXPECT_EQ(tensor_to_fill.type(), dtype)
        << "Fill: Result data type mismatch";
    EXPECT_THAT(tensor_to_fill.shape(), ::testing::ElementsAreArray({3u, 2u}))
        << "Fill: Result shape mismatch";

    std::vector<Scalar> expected_values(6, fill_value);
    EXPECT_THAT(
        getTenData(tensor_to_fill),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
        << "Fill: Result data mismatch";
  }

  {
    Tensor tensor_to_zero = tensor_f32_B;
    tensor_to_zero.fill(0.0);

    EXPECT_EQ(tensor_to_zero.type(), dtype)
        << "Fill zero: Result data type mismatch";
    EXPECT_THAT(tensor_to_zero.shape(), ::testing::ElementsAreArray({3u, 2u}))
        << "Fill zero: Result shape mismatch";

    std::vector<Scalar> expected_zeros(6, 0.0);
    EXPECT_THAT(
        getTenData(tensor_to_zero),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_zeros))
        << "Fill zero: Result data mismatch";
  }

  {
    Tensor tensor_negative_fill =
        Tensor(std::vector<float>{10.0f, 20.0f}, std::vector<size_t>{1u, 2u},
               device, dtype);

    Scalar negative_value = -99.0;
    tensor_negative_fill.fill(negative_value);

    EXPECT_EQ(tensor_negative_fill.type(), dtype)
        << "Fill negative: Result data type mismatch";
    EXPECT_THAT(tensor_negative_fill.shape(),
                ::testing::ElementsAreArray({1u, 2u}))
        << "Fill negative: Result shape mismatch";

    std::vector<Scalar> expected_negative(2, negative_value);
    EXPECT_THAT(
        getTenData(tensor_negative_fill),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_negative))
        << "Fill negative: Result data mismatch";
  }

  {
    Tensor scalar_fill_test = scalar_tensor_f32;
    scalar_fill_test.fill(777.0);

    EXPECT_EQ(scalar_fill_test.type(), dtype)
        << "Fill scalar tensor: Result data type mismatch";
    EXPECT_THAT(scalar_fill_test.shape(), ::testing::ElementsAreArray({1}))
        << "Fill scalar tensor: Result shape mismatch";

    std::vector<Scalar> expected_scalar_value = {777.0};
    EXPECT_THAT(
        getTenData(scalar_fill_test),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_scalar_value))
        << "Fill scalar tensor: Result data mismatch";
  }

  {
    Tensor tensor_3d_fill =
        tensor_f32_1x2x1_squeeze_expand;  // Copy constructor
    Scalar fill_3d_value = 123.0;
    tensor_3d_fill.fill(fill_3d_value);

    EXPECT_EQ(tensor_3d_fill.type(), dtype)
        << "Fill 3D: Result data type mismatch";
    EXPECT_THAT(tensor_3d_fill.shape(),
                ::testing::ElementsAreArray({1u, 2u, 1u}))
        << "Fill 3D: Result shape mismatch";

    std::vector<Scalar> expected_3d_values(2, fill_3d_value);
    EXPECT_THAT(
        getTenData(tensor_3d_fill),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_3d_values))
        << "Fill 3D: Result data mismatch";
  }
}

namespace {

std::vector<ParamTypes> GenerateParamCombinations() {
  std::vector<ParamTypes> comb;
  for (auto dtype : {DataType::Int16, DataType::Int32, DataType::Int64,
                     DataType::Float32, DataType::Float64}) {
    for (auto device : {DeviceType::CPU, DeviceType::CUDA}) {
      comb.push_back(std::make_tuple(device, dtype));
    }
  }
  return comb;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(TensorOp, TensorOpTest,
                         testing::ValuesIn(GenerateParamCombinations()));

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
