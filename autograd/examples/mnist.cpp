#include "mnist.h"
#include <cassert>
#include <iostream>
#include "autograd/engine/tensor_ops.h"
#include "autograd/engine/constructor.h"
#include "autograd/engine/variable.h"
#include "autograd/engine/variable_ops.h"
#include "autograd/util/csv_reader.h"
#include <algorithm>   
#include <iterator>    
#include <random>      
#include <vector>

using autograd::Tensor;
using autograd::Variable;
using autograd::rand;
using autograd::tensor;

namespace {

template <typename T>
std::vector<float> flatten(const std::vector<std::vector<T>>& vec2d) {
  std::vector<T> flattened;

  size_t total_size = 0;
  for (const auto& row : vec2d) {
    total_size += row.size();
  }
  flattened.reserve(total_size);

  for (const auto& row : vec2d) {
    flattened.insert(flattened.end(), row.begin(), row.end());
  }
  return flattened;
}

template<typename X, typename Y>
void sample_batch_sample(
    const std::vector<X>& data,
    const std::vector<Y>& labels,
    std::size_t k,
    std::vector<X>& out_data,
    std::vector<Y>& out_labels
) {
    assert(data.size() == labels.size());
    std::size_t n = data.size();
    k = std::min(k, n);

    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0U);

    std::vector<std::size_t> pick;
    pick.reserve(k);
    std::sample(
        idx.begin(), idx.end(),
        std::back_inserter(pick),
        k,
        std::mt19937{std::random_device{}()}
    );

    out_data.reserve(k);
    out_labels.reserve(k);
    for (auto i : pick) {
        out_data.push_back(data[i]);
        out_labels.push_back(labels[i]);
    }
}

float calculate_accuracy(const Variable& weights1, const Variable& weights2, 
                        const std::vector<std::vector<float>>& data, 
                        const std::vector<std::vector<float>>& labels) {
  int batch_size = data.size();

  Variable target_inputs = tensor(data, false);
  Variable target_labels = tensor(labels, false);
  
  Variable a1 = target_inputs.matmul(weights1);
  Variable z1 = a1.relu();
  Variable a2 = z1.matmul(weights2);
  
  Variable exp = a2.exp();
  Variable exp_tmp = a2.exp();
  std::vector<float> broad_data(10, 1);
  Variable broad(Tensor(broad_data, {1, 10}), false);
  Variable denom = exp_tmp.sum(1).matmul(broad) + 1e-10F;
  
  Variable z2 = exp / denom;
  
  Variable true_scores = (z2 * target_labels).sum(1);
  Variable max_scores = z2.max(1);
  Variable correct = (true_scores == max_scores).sum(0).sum(1) / batch_size;
  
  return correct.data().data[0];
}

}  // namespace

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
    sample_batch_sample(train_data, train_label, batch_size, out_data, out_labels);

    Variable inputs = tensor(out_data, false);
    Variable labels = tensor(out_labels, false);

    Variable a1 = inputs.matmul(weights1);
    Variable z1 = a1.relu();
    Variable a2 = z1.matmul(weights2);  // 1200 x 10

    Variable exp = a2.exp();
    Variable exp_tmp = a2.exp();  // 1200 x 1
    std::vector<float> broad_data(10, 1);
    Variable broad(Tensor(broad_data, {1, 10}), false);
    Variable denom = exp_tmp.sum(1).matmul(broad) + 1e-10F;  // 1200 x 10
    Variable z2 = exp / denom;

    Variable loss =
        ((-1.0F) * z2.log() * labels).sum(0).sum(1) / 1200;
    loss.backward();

    if (i % 100 == 0) {
      Variable true_scores = (z2 * labels).sum(1); // 128 x 1
      Variable max_scores = z2.max(1);
      Variable correct = (true_scores == max_scores).sum(0).sum(1) / batch_size;
      std::cout << "Iteration " << i << " - Training accuracy: " << correct.data().data[0] << std::endl;
    }
    
    if (i % 10000 == 0) {
      int test_batch_size = std::min(static_cast<int>(test_data.size()), 1000);
      std::vector<std::vector<float>> test_batch_data;
      std::vector<std::vector<float>> test_batch_labels;
      sample_batch_sample(test_data, test_label, test_batch_size, test_batch_data, test_batch_labels);
      
      float test_accuracy = calculate_accuracy(weights1, weights2, test_batch_data, test_batch_labels);
      std::cout << "Iteration " << i << " - Validation accuracy: " << test_accuracy << std::endl;
    }

    float learning_rate = 0.01;
    weights1 =
        Variable(weights1.data() - learning_rate * weights1.grad(), true);
    weights2 =
        Variable(weights2.data() - learning_rate * weights2.grad(), true);
  }
}