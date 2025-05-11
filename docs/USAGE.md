```cpp
#include "mnist.h"
#include <cassert>
#include <iostream>
#include "include/lamppp/tensor/constructor.h"
#include "include/lamppp/tensor/variable.h"
#include "include/lamppp/tensor/variable_ops.h"
#include "autograd/util/csv_reader.h"
#include "autograd/util/batch_sample.h"
#include <algorithm>
#include <vector>

using autograd::Variable;
using autograd::rand;
using autograd::tensor;
using autograd::ones;

int main() {
  auto [train_data, train_label] = readCSV("data/mnist_train.csv");
  auto [test_data, test_label] = readCSV("data/mnist_test.csv");

  Variable weights1 = rand({784, 256}, true) / 100;
  Variable weights2 = rand({256, 10}, true) / 100;

  int epochs = 1e9;
  for (int i = 0; i < epochs; i++) {
    const auto forward = [&](const Variable& xs) {
      Variable a1 = xs.matmul(weights1);
      Variable z1 = a1.relu();
      Variable a2 = z1.matmul(weights2);  // 1200 x 10

      Variable exp = a2.exp();
      Variable denom = exp.sum(1).matmul(ones({1, 10}, false)) + 1e-10F;
      Variable z2 = exp / denom;

      return z2;
    };

    Variable inputs = tensor(train_data, false);
    Variable labels = tensor(test_data, false);

    Variable out_layer = forward(inputs);
    Variable loss = ((-1.0F) * out_layer.log() * labels).sum(0).sum(1) / 1200;
    loss.backward();

    float learning_rate = 0.01;
    weights1 =
        Variable(weights1.data() - learning_rate * weights1.grad(), true);
    weights2 =
        Variable(weights2.data() - learning_rate * weights2.grad(), true);
  }
}

```
