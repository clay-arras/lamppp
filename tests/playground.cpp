#include <vector>
#include "lamppp/lamppp.hpp"

int main() {
  auto a = lmp::tensor::Tensor(std::vector<int>{1, 2, 3, 4, 5, 2},
                               std::vector<size_t>{3, 2});

  auto b = lmp::tensor::ops::sum(a, 1);
  std::cout << b << std::endl;
}