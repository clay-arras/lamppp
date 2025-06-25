#include <cuda_runtime_api.h>
#include "lamppp/lamppp.hpp"

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
}