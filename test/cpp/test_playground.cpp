#include <algorithm>
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/tensor.hpp"
#include "autograd/engine/tensor_helper.hpp"

int main() {
  // std::vector<autograd::Scalar> data1 = {1.0, 2.0, -1.0};
  // std::vector<size_t> shape1 = {1, 3};
  // autograd::Tensor tensor_data1 =
  //     autograd::Tensor(data1, shape1, DeviceType::CUDA, DataType::Float32);

  // std::vector<autograd::Scalar> data2 = {1.0f, 2.0f, 3.0f};
  // std::vector<size_t> shape2 = {1, 3};
  // autograd::Tensor tensor_data2 =
  //     autograd::Tensor(data2, shape2, DeviceType::CUDA, DataType::Float32);

  // autograd::Variable variable_data1(tensor_data1, true);
  // autograd::Variable variable_data2(tensor_data2, true);

  // autograd::Tensor result = tensor_data1 + 10;
  // std::cout << "Result: " << result << std::endl;
  // std::cout << "Tensor Data 1: " << tensor_data1 << std::endl;

  std::vector<float> vec1(100000);
  std::vector<float> vec2(100000);
  std::generate(vec1.begin(), vec1.end(),
                []() { return static_cast<float>(rand()) / RAND_MAX; });
  std::generate(vec2.begin(), vec2.end(),
                []() { return static_cast<float>(rand()) / RAND_MAX; });

  std::vector<float> result(100000);
  std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(),
                 std::plus<float>());

  std::vector<size_t> shape = {10, 50, 2};
  autograd::Tensor tensor_data1 =
      autograd::Tensor(vec1, shape, DeviceType::CUDA, DataType::Float64);
  autograd::Tensor tensor_data2 =
      autograd::Tensor(vec2, shape, DeviceType::CUDA, DataType::Float64);

  autograd::Tensor result_ten = tensor_data1 + tensor_data2;

  std::span<float> ok = result_ten.view<float>();

  bool are_equal = std::equal(ok.begin(), ok.end(), result.begin());
  if (are_equal) {
    std::cout << "The tensors are equal." << std::endl;
  } else {
    std::cout << "The tensors are not equal." << std::endl;
  }
}