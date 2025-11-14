#include <cassert>
#include <functional>
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

      lmp::Variable test_inputs = lmp::autograd::tensor(test_batch_data, false);
      lmp::Variable test_labels =
          lmp::autograd::tensor(test_batch_labels, false);
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

    for (std::reference_wrapper<lmp::nets::Parameter> params : model.parameters()) {
      params.get() = lmp::nets::Parameter(
          lmp::Variable(params.get().data() - kLearningRate * params.get().grad(), true));
    }
  }
}
