#include <vector>
#include "lamppp/lamppp.hpp"
#include "lamppp/tensor/device_type.hpp"

int main() {
  auto a = lmp::tensor::Tensor(std::vector<int>{1, 2, 3, 4, 5, 2},
                               std::vector<size_t>{3, 2},
                               lmp::tensor::DeviceType::CUDA, 
                               lmp::tensor::DataType::Float32);

  std::cout << a << std::endl;

  auto b = a.to(lmp::tensor::DeviceType::CPU);

  std::cout << b << std::endl;
}