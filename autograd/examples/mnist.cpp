#include "mnist.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "autograd/engine/constructor.h"
#include "autograd/engine/tensor_ops.h"
#include "autograd/engine/variable.h"
#include "autograd/engine/variable_ops.h"
#include "autograd/util/batch_sample.h"
#include "autograd/util/csv_reader.h"

using autograd::ones;
using autograd::rand;
using autograd::tensor;
using autograd::Variable;

int main() {
  auto [train_data, train_label] = readCSV("data/mnist_train.csv");
  auto [test_data, test_label] = readCSV("data/mnist_test.csv");

  Variable weights1 = rand({784, 256}, true) / 100;
  Variable weights2 = rand({256, 10}, true) / 100;

  int epochs = 1e9;
  int batch_size = 128;

  for (int i = 0; i < epochs; i++) {
    std::vector<std::vector<float>> out_data;
    std::vector<std::vector<float>> out_labels;
    sample_batch_sample(train_data, train_label, batch_size, out_data,
                        out_labels);

    const auto forward = [&](const Variable& xs) {
      Variable a1 = xs.matmul(weights1);
      Variable z1 = a1.relu();
      Variable a2 = z1.matmul(weights2);  // 1200 x 10

      Variable exp = a2.exp();
      Variable denom = exp.sum(1).matmul(ones({1, 10}, false)) + 1e-10F;
      Variable z2 = exp / denom;

      return z2;
    };

    Variable inputs = tensor(out_data, false);
    Variable labels = tensor(out_labels, false);

    Variable out_layer = forward(inputs);
    Variable loss = ((-1.0F) * out_layer.log() * labels).sum(0).sum(1) / 1200;
    loss.backward();

    if (i % 100 == 0) {
      Variable true_scores = (out_layer * labels).sum(1);  // 128 x 1
      Variable max_scores = out_layer.max(1);
      Variable correct = (true_scores == max_scores).sum(0).sum(1) / batch_size;
      std::cout << "Iteration " << i
                << " - Training accuracy: " << correct.data().data()[0]
                << std::endl;
    }

    if (i % 10000 == 0) {
      int test_batch_size = std::min(static_cast<int>(test_data.size()), 1000);
      std::vector<std::vector<float>> test_batch_data;
      std::vector<std::vector<float>> test_batch_labels;
      sample_batch_sample(test_data, test_label, test_batch_size,
                          test_batch_data, test_batch_labels);

      Variable test_inputs = tensor(test_batch_data, false);
      Variable test_labels = tensor(test_batch_labels, false);
      Variable test_out_layer = forward(test_inputs);

      Variable true_scores = (test_out_layer * test_labels).sum(1);
      Variable max_scores = test_out_layer.max(1);
      Variable correct = (true_scores == max_scores).sum(0).sum(1) /
                         static_cast<float>(test_batch_size);
      float test_accuracy = correct.data().data()[0];

      std::cout << "Iteration " << i
                << " - Validation accuracy: " << test_accuracy << std::endl;
    }

    float learning_rate = 0.01;
    weights1 =
        Variable(weights1.data() - learning_rate * weights1.grad(), true);
    weights2 =
        Variable(weights2.data() - learning_rate * weights2.grad(), true);
  }
}