#include "lamppp/autograd/constructor.hpp"
#include "lamppp/lamppp.hpp"
#include "lamppp/nets/layers/linear.hpp"

int main() {
  lmp::Variable input = lmp::autograd::rand({32, 1024}, true);

  lmp::nets::Linear layer({1024, 512});
  std::cout << input << std::endl;
  lmp::Variable output = layer(input);
  std::cout << output << std::endl;
}