#include <algorithm>
#include <vector>
#include "include/lamppp/autograd/variable.hpp"
#include "include/lamppp/autograd/variable_ops.hpp"
#include "include/lamppp/tensor/data_type.hpp"
#include "include/lamppp/tensor/device_type.hpp"
#include "include/lamppp/tensor/tensor.hpp"
#include "include/lamppp/tensor/tensor_helper.hpp"

int main() {
  std::vector<float> vec1(10000);
  std::vector<float> vec2(10000);
  std::generate(vec1.begin(), vec1.end(),
                []() { return static_cast<float>(rand()) / RAND_MAX; });
  std::generate(vec2.begin(), vec2.end(),
                []() { return static_cast<float>(rand()) / RAND_MAX; });

  std::vector<float> result(10000);
  std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(),
                 std::plus<float>());

  std::vector<size_t> shape = {10, 500, 2};
  autograd::Tensor tensor_data1 =
      autograd::Tensor(vec1, shape, DeviceType::CUDA, DataType::Float64);
  autograd::Tensor tensor_data2 =
      autograd::Tensor(vec2, shape, DeviceType::CUDA, DataType::Float64);

  autograd::Variable variable1 = autograd::Variable(tensor_data1);
  autograd::Variable variable2 = autograd::Variable(tensor_data1);

  // autograd::Tensor result_ten = tensor_data1 + tensor_data2;
  // std::span<float> ok = result_ten.view<float>();
  autograd::Variable result_var = variable1 + variable2;
  std::span<float> ok = result_var.data().view<float>();

  bool are_equal = std::equal(ok.begin(), ok.end(), result.begin());
  if (are_equal) {
    std::cout << "The tensors are equal." << std::endl;
  } else {
    std::cout << "The tensors are not equal." << std::endl;
  }
}