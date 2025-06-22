#include <any>
#include "lamppp/autograd/constructor.hpp"
#include "lamppp/lamppp.hpp"
#include "lamppp/nets/any.hpp"
#include "lamppp/nets/layers/activation.hpp"
#include "lamppp/nets/layers/container.hpp"
#include "lamppp/nets/layers/linear.hpp"

int main() {
  lmp::Variable input = lmp::autograd::rand({32, 1024}, true);

  lmp::nets::Linear layer1(1024, 512);
  lmp::nets::ReLU layer2;
  lmp::nets::Linear layer3(512, 128);
  lmp::nets::ReLU layer4;

  std::vector<lmp::nets::AnyModule> layers = {
      lmp::nets::AnyModule(layer1), lmp::nets::AnyModule(layer2),
      lmp::nets::AnyModule(layer3), lmp::nets::AnyModule(layer4)};
  lmp::nets::Sequential model(layers);
  auto output = std::any_cast<lmp::Variable>(
      model(std::vector<std::any>{std::any(input)}));
  std::cout << "Out: " << output << std::endl;

  for (const auto& params : model.named_parameters()) {
    std::cout << params.first << " " << params.second << std::endl;
  }
}