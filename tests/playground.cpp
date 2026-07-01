#include <functional>
#include "lamp3/lamp3.hpp"
#include "lamp3/nets/layers/linear.hpp"
#include "lamp3/nets/parameter.hpp"
#include "lamp3/tensor/native/conv_ops.hpp"

// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
  auto input = lmp::tensor::Tensor(
      std::vector<float>{1, -4, 2, 7, 9, 10, 4, -2, 27, 0, 0, 24, 2,
                         2, 10, 1, 4, 0, 2,  1, 1,  4,  0, 2, 1},
      std::vector<size_t>{5, 5}, lmp::DeviceType::CUDA);

  auto kernel =
      lmp::tensor::Tensor(std::vector<float>{0, 1, -1, 1, 3, 0, 1, -3, 0},
                          std::vector<size_t>{3, 3}, lmp::DeviceType::CUDA);

  lmp::Tensor output = lmp::tensor::ops::conv2d(input, kernel, 2, 4, 2);
  std::cout << output << std::endl;
}