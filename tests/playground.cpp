#include <vector>
#include "include/lamppp/lamppp.hpp"

int main() {
  auto a =
      lmp::tensor::Tensor(std::vector<int>{1, 2, 3}, std::vector<size_t>{3, 1});
  auto b = lmp::tensor::Tensor(std::vector<int>{4, 5}, std::vector<size_t>{2});

  auto c = a + b;
  std::cout << c << std::endl;
}