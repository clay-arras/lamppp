#include <cassert>
#include <iostream>
#include <limits>
#include <vector>
#include "lamppp/lamppp.hpp"
#include "utils/batch_sample.hpp"
#include "utils/csv_reader.hpp"

int main() {
  auto [train_data, train_label] = readCSV("data/mnist_train.csv");
  auto [test_data, test_label] = readCSV("data/mnist_test.csv");

  lmp::Variable weights1 =
      lmp::autograd::rand({784, 256}, lmp::DeviceType::CUDA,
                          lmp::DataType::Float32, true) /
      100;
  lmp::Variable weights2 = lmp::autograd::rand({256, 10}, lmp::DeviceType::CUDA,
                                               lmp::DataType::Float32, true) /
                           100;

  int epochs = 1e9;
  int batch_size = 128;

  for (int i = 0; i < epochs; i++) {
    std::vector<std::vector<float>> out_data;
    std::vector<std::vector<float>> out_labels;
    sample_batch_sample(train_data, train_label, batch_size, out_data,
                        out_labels);

    const auto forward = [&](const lmp::Variable& xs) {
      lmp::Variable a1 = lmp::matmul(xs, weights1);
      lmp::Variable z1 = lmp::clamp(
          a1, 0.0, std::numeric_limits<float>::max());  
      lmp::Variable a2 = lmp::matmul(z1, weights2);     // 1200 x 10

      lmp::Variable exp = lmp::exp(a2);
      lmp::Variable denom =
          lmp::matmul(lmp::sum(exp, 1),
                      lmp::autograd::ones({1, 10}, lmp::DeviceType::CUDA,
                                          lmp::DataType::Float32, false)) +
          1e-10F;
      lmp::Variable z2 = exp / denom;

      return z2;
    };

    lmp::Variable inputs = lmp::autograd::tensor(
        out_data, lmp::DeviceType::CUDA, lmp::DataType::Float32, false);
    lmp::Variable labels = lmp::autograd::tensor(
        out_labels, lmp::DeviceType::CUDA, lmp::DataType::Float32, false);

    lmp::Variable out_layer = forward(inputs);
    lmp::Variable loss =
        lmp::sum(lmp::sum((-lmp::log(out_layer) * labels), 0), 1) /
        1200;
    loss.backward();

    if (i % 100 == 0) {
      lmp::Variable true_scores = lmp::sum((out_layer * labels), 1);
      lmp::Variable max_scores = lmp::max(out_layer, 1);
      lmp::Variable correct =
          lmp::sum(lmp::sum((true_scores == max_scores), 0), 1) /
          static_cast<float>(batch_size);
      std::cout << "Iteration " << i << " - Training accuracy: "
                << correct.data().to_vector<float>()[0] << std::endl;
    }

    if (i % 10000 == 0) {
      int test_batch_size = std::min(static_cast<size_t>(test_data.size()),
                                     static_cast<size_t>(1000));
      std::vector<std::vector<float>> test_batch_data;
      std::vector<std::vector<float>> test_batch_labels;
      sample_batch_sample(test_data, test_label, test_batch_size,
                          test_batch_data, test_batch_labels);

      lmp::Variable test_inputs =
          lmp::autograd::tensor(test_batch_data, lmp::DeviceType::CUDA,
                                lmp::DataType::Float32, false);
      lmp::Variable test_labels =
          lmp::autograd::tensor(test_batch_labels, lmp::DeviceType::CUDA,
                                lmp::DataType::Float32, false);
      lmp::Variable test_out_layer = forward(test_inputs);

      lmp::Variable true_scores = lmp::sum((test_out_layer * test_labels), 1);
      lmp::Variable max_scores = lmp::max(test_out_layer, 1);
      lmp::Variable correct =
          lmp::sum(lmp::sum((true_scores == max_scores), 0), 1) /
          static_cast<float>(test_batch_size);
      float test_accuracy = correct.data().to_vector<float>()[0];

      std::cout << "Iteration " << i
                << " - Validation accuracy: " << test_accuracy << std::endl;
    }

    float learning_rate = 0.01;
    weights1 =
        lmp::Variable(weights1.data() - learning_rate * weights1.grad(), true);
    weights2 =
        lmp::Variable(weights2.data() - learning_rate * weights2.grad(), true);
  }
}