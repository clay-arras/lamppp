#include <any>
#include "lamppp/autograd/constructor.hpp"
#include "lamppp/autograd/grad_utils.hpp"
#include "lamppp/autograd/functions/matrix_ops.hpp"
#include "lamppp/lamppp.hpp"
#include "lamppp/nets/any.hpp"
#include "lamppp/nets/layers/activation.hpp"
#include "lamppp/nets/layers/container.hpp"
#include "lamppp/nets/layers/linear.hpp"

int main() {
  lmp::Variable input = lmp::autograd::rand({4, 2}, true);
  // lmp::Variable weights = lmp::autograd::randn(0, 1, {1024, 512});
  // lmp::Variable bias = lmp::autograd::randn(0, 1, {512});

  lmp::nets::Linear layer1(2, 3);
  auto output = layer1(input);
  
  output.backward();
  std::cout << "Out: " << output << std::endl;

  // for (const auto& params : model.named_parameters()) {
  //   std::cout << params.first << " " << params.second << std::endl;
  // }

  // lmp::Variable output = lmp::autograd::ops::matmul(input, weights) + bias;
  // output.backward();


  // lmp::Tensor ten = lmp::autograd::ones({4, 3}).data();
  // auto grad = lmp::autograd::detail::sum_broadcast_axis(ten, {3});
  // std::cout << grad << std::endl;
}