#include <vector>
#include "lamppp/lamppp.hpp"
#include "lamppp/tensor/device_type.hpp"

int main() {
  auto a = lmp::tensor::Tensor(std::vector<int>{1, 2, 3, 4, 5, 2},
                               std::vector<size_t>{3, 2},
                               lmp::tensor::DeviceType::CPU);

  auto c = a.to_vector<int>();
  for (const auto& elem : c) {
    std::cout << elem << " ";
  }
  std::cout << std::endl;

  std::cout << a << std::endl;
  auto b = lmp::tensor::ops::exp(a);
  std::cout << b << std::endl;
}