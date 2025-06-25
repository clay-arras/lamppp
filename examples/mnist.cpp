#include <cassert>
#include <iostream>
#include <limits>
#include <vector>
#include "lamppp/lamppp.hpp"
#include "lamppp/nets/any.hpp"
#include "lamppp/nets/layers/activation.hpp"
#include "lamppp/nets/layers/container.hpp"
#include "lamppp/nets/layers/linear.hpp"
#include "lamppp/nets/parameter.hpp"
#include "utils/batch_sample.hpp"
#include "utils/csv_reader.hpp"


const int kEpochs = 1e9;
const int kBatchSize = 128;
const float kLearningRate = 0.01;

int main() {
  auto [train_data, train_label] = readCSV("data/mnist_train.csv");
  auto [test_data, test_label] = readCSV("data/mnist_test.csv");

  lmp::nets::Sequential model(std::vector<lmp::nets::AnyModule>{
      lmp::nets::AnyModule(lmp::nets::Linear(784, 512, false)),
      lmp::nets::AnyModule(lmp::nets::ReLU()),
      lmp::nets::AnyModule(lmp::nets::Linear(512, 10, false)),
      lmp::nets::AnyModule(lmp::nets::Softmax(-1))
  });

  for (int i = 0; i < kEpochs; i++) {
    std::vector<std::vector<float>> out_data;
    std::vector<std::vector<float>> out_labels;
    sample_batch_sample(train_data, train_label, kBatchSize, out_data,
                        out_labels);

    lmp::Variable inputs = lmp::autograd::tensor(out_data, false);
    lmp::Variable labels = lmp::autograd::tensor(out_labels, false);

    auto out_layer = std::any_cast<lmp::Variable>(
        model(std::vector<std::any>{std::any(inputs)}));

    lmp::Variable loss =
        lmp::sum(lmp::sum((-lmp::log(out_layer + 0.001) * labels), 0), 1) / 1200;
    loss.backward();
    
    if (i % 100 == 0) {
      lmp::Variable true_scores = lmp::sum((out_layer * labels), 1);
      lmp::Variable max_scores = lmp::max(out_layer, 1);
      lmp::Variable correct =
          lmp::sum(lmp::sum((true_scores == max_scores), 0), 1) /
          static_cast<float>(kBatchSize);
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
          lmp::autograd::tensor(test_batch_data, false, lmp::DeviceType::CUDA,
                                lmp::DataType::Float32);
      lmp::Variable test_labels =
          lmp::autograd::tensor(test_batch_labels, false, lmp::DeviceType::CUDA,
                                lmp::DataType::Float32);
      auto test_out_layer = std::any_cast<lmp::Variable>(
          model(std::vector<std::any>{std::any(test_inputs)}));

      lmp::Variable true_scores = lmp::sum((test_out_layer * test_labels), 1);
      lmp::Variable max_scores = lmp::max(test_out_layer, 1);
      lmp::Variable correct =
          lmp::sum(lmp::sum((true_scores == max_scores), 0), 1) /
          static_cast<float>(test_batch_size);
      float test_accuracy = correct.data().to_vector<float>()[0];

      std::cout << "Iteration " << i
                << " - Validation accuracy: " << test_accuracy << std::endl;
    }

    for (const auto& params : model.parameters()) {
    //   params = lmp::nets::Parameter(
    //       lmp::Variable(params.data() - kLearningRate * params.grad(), true));
    }
  }
}

// #include <cuda_runtime_api.h>
// #include <cassert>
// #include <iostream>
// #include <limits>
// #include <vector>
// #include "lamppp/lamppp.hpp"
// #include "utils/batch_sample.hpp"
// #include "utils/csv_reader.hpp"

// namespace {

// lmp::Variable forward(const lmp::Variable& xs, const lmp::Variable& weights1, const lmp::Variable& weights2) {
//     lmp::Variable a1 = lmp::matmul(xs, weights1);
//     lmp::Variable z1 = lmp::clamp(
//         a1, 0.0, std::numeric_limits<float>::max());  
//     lmp::Variable a2 = lmp::matmul(z1, weights2);

//     lmp::Variable exp = lmp::exp(a2);
//     lmp::Variable denom =
//         lmp::matmul(lmp::sum(exp, 1),
//                     lmp::autograd::ones({1, 10}, false, lmp::DeviceType::CUDA,
//                                         lmp::DataType::Float32)) +
//         1e-10F;
//     lmp::Variable z2 = exp / denom;

//     return z2;
// };

// }

// int main() {
//   auto [train_data, train_label] = readCSV("data/mnist_train.csv");
//   auto [test_data, test_label] = readCSV("data/mnist_test.csv");

//   lmp::Variable weights1 = lmp::autograd::randn(
//       0, 0.01, {784, 256}, true, lmp::DeviceType::CUDA, lmp::DataType::Float32);
//   lmp::Variable weights2 = lmp::autograd::randn(
//       0, 0.01, {256, 10}, true, lmp::DeviceType::CUDA, lmp::DataType::Float32);

//   int epochs = 1e9;
//   int batch_size = 128;

//   for (int i = 0; i < epochs; i++) {
//     std::vector<std::vector<float>> out_data;
//     std::vector<std::vector<float>> out_labels;
//     sample_batch_sample(train_data, train_label, batch_size, out_data,
//                         out_labels);

//     lmp::Variable inputs = lmp::autograd::tensor(
//         out_data, false, lmp::DeviceType::CUDA, lmp::DataType::Float32);
//     lmp::Variable labels = lmp::autograd::tensor(
//         out_labels, false, lmp::DeviceType::CUDA, lmp::DataType::Float32);

//     lmp::Variable out_layer = forward(inputs, weights1, weights2);
//     lmp::Variable loss =
//         lmp::sum(lmp::sum((-lmp::log(out_layer) * labels), 0), 1) /
//         1200;
//     loss.backward();

//     if (i % 100 == 0) {
//         lmp::Variable true_scores = lmp::sum((out_layer * labels), 1);
//         lmp::Variable max_scores = lmp::max(out_layer, 1);
//         lmp::Variable correct =
//             lmp::sum(lmp::sum((true_scores == max_scores), 0), 1) /
//             static_cast<float>(batch_size);
//         std::cout << "Iteration " << i << " - Training accuracy: "
//                     << correct.data().to_vector<float>()[0] << std::endl;
//         }

//         if (i % 10000 == 0) {
//         int test_batch_size = std::min(static_cast<size_t>(test_data.size()),
//                                         static_cast<size_t>(1000));
//         std::vector<std::vector<float>> test_batch_data;
//         std::vector<std::vector<float>> test_batch_labels;
//         sample_batch_sample(test_data, test_label, test_batch_size,
//                             test_batch_data, test_batch_labels);

//         lmp::Variable test_inputs =
//             lmp::autograd::tensor(test_batch_data, false, lmp::DeviceType::CUDA,
//                                     lmp::DataType::Float32);
//         lmp::Variable test_labels =
//             lmp::autograd::tensor(test_batch_labels, false, lmp::DeviceType::CUDA,
//                                     lmp::DataType::Float32);
//         lmp::Variable test_out_layer = forward(test_inputs, weights1, weights2);

//         lmp::Variable true_scores = lmp::sum((test_out_layer * test_labels), 1);
//         lmp::Variable max_scores = lmp::max(test_out_layer, 1);
//         lmp::Variable correct =
//             lmp::sum(lmp::sum((true_scores == max_scores), 0), 1) /
//             static_cast<float>(test_batch_size);
//         float test_accuracy = correct.data().to_vector<float>()[0];

//         std::cout << "Iteration " << i
//                     << " - Validation accuracy: " << test_accuracy << std::endl;
//         }

//     float learning_rate = 0.01;
//     weights1 =
//         lmp::Variable(weights1.data() - learning_rate * weights1.grad(), true);
//     weights2 =
//         lmp::Variable(weights2.data() - learning_rate * weights2.grad(), true);
//   }
// }