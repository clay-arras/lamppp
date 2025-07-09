#include <cuda_runtime_api.h>
#include <functional>
#include "lamppp/lamppp.hpp"
#include "lamppp/nets/layers/linear.hpp"
#include "lamppp/nets/parameter.hpp"

int main() {
  lmp::Variable weights1 = lmp::autograd::rand({784, 256}, true);

  for (int i = 0; i < 10000; i++) {
    lmp::Variable weights2 = lmp::autograd::rand({256, 10}, true);
    lmp::Variable loss = lmp::matmul(weights1, weights2);
    loss.backward();

    float learning_rate = 0.01;
    weights1 = lmp::Variable(weights1.data() - learning_rate * weights1.grad(),
                             true);  // grad_fn is cleared
  }

  lmp::nets::Linear layer(20, 10);
  for (int i = 0; i < 5; i++) {
    for (std::reference_wrapper<lmp::nets::Parameter> params :
         layer.parameters()) {
      std::cout << params.get() << std::endl;
      params.get() = lmp::nets::Parameter(params.get() - 1);
    }
  }
}