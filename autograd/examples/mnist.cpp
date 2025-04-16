#include "mnist.h"
#include "autograd/util/csv_reader.h"
#include <cassert>
#include <iostream>
#include "autograd/engine/variable.h"
#include "autograd/engine/variable_ops.h"
#include "autograd/engine/tensor_ops.h"
#include "autograd/engine/functions/matrix_ops.h"
#include "autograd/engine/functions/reduct_ops.h"

using autograd::Variable;
using autograd::Tensor;

namespace {

template<typename T>
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

}

int main() {
  auto [data, label] = readCSV("data/mnist_dummy.csv");

  std::vector<float> w1_data;
  w1_data.reserve(784 * 256);
  for (int i = 0; i < 784 * 256; i++) {
    w1_data.push_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.01F);
  }
  std::vector<float> w2_data;
  w2_data.reserve(256 * 10);
  for (int i = 0; i < 256 * 10; i++) {
    w2_data.push_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.01F);
  }

  Variable weights1(Tensor(w1_data, {784, 256}), true);
  Variable weights2(Tensor(w2_data, {256, 10}), true);

  std::vector<float> flattened_data = flatten(data);
  Variable inputs(Tensor(flattened_data, {784, 1200}), true);

  std::vector<float> flattened_labels = flatten(label);
  Variable labels(Tensor(flattened_labels, {10, 1200}), true); // IS THIS SHAPE RIGHT


  for (int i=0; i<10; i++) {

    Variable a1 = inputs.transpose().matmul(weights1);
    Variable z1 = a1.relu();
    Variable a2 = z1.matmul(weights2); // 1200 x 10

    Variable exp = a2.exp();
    std::vector<float> broad_data(10, 1);
    Variable broad(Tensor(broad_data, {1, 10}), false);
    Variable denom = exp.sum(1).matmul(broad) + 1e-10F; // 1200 x 10

    Variable z2 = exp / denom;

    Variable loss = (-1.0F) * (z2.log() * labels.transpose()).sum(0).sum(1);
    std::cout << loss << std::endl;
    loss.backward();

    float learning_rate = 0.0001;
    weights1 = Variable(weights1.data() + learning_rate * weights1.grad(), true);
    weights2 = Variable(weights2.data() + learning_rate * weights2.grad(), true);

  }
}