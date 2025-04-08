#include <benchmark/benchmark.h>
#include <memory>
#include <vector>
#include "autograd/engine/engine.h"
#include "autograd/nn/nn.h"
#include "autograd/util/csv_reader.h"

using std::vector;

namespace {

/**
 * @brief Computes the softmax of a vector of shared values.
 *
 * This function takes a vector of shared pointers to Value objects, 
 * computes the softmax activation, and normalizes the values to 
 * represent a probability distribution. It asserts that the input 
 * vector has exactly 10 elements.
 *
 * @param x A vector of shared pointers to Value objects.
 * @return A vector of shared pointers to Value objects after applying softmax.
 */
std::vector<std::shared_ptr<Value>> softmax(
    std::vector<std::shared_ptr<Value>> x) {
  assert((int)x.size() == 10);
  std::shared_ptr<Value> denom = std::make_shared<Value>(Value(1e-4));
  for (const auto& i : x)
    denom = denom + i->exp();
  for (auto& i : x)
    i = i->exp() / denom;
  return x;
};

/**
 * @brief Performs a forward pass through two layers.
 *
 * This function takes an input vector of shared pointers to Value objects, 
 * applies the first layer, and then the second layer, followed by a softmax 
 * activation. The output is a vector of shared pointers to Value objects.
 *
 * @param x A constant reference to a vector of shared pointers to Value objects.
 * @param w1 The first layer to apply.
 * @param w2 The second layer to apply.
 * @return A vector of shared pointers to Value objects after the forward pass.
 */
std::vector<std::shared_ptr<Value>> forward(
    const std::vector<std::shared_ptr<Value>>& x, Layer& w1, Layer& w2) {
  std::vector<std::shared_ptr<Value>> z1 = w1(std::move(x));
  std::vector<std::shared_ptr<Value>> z2 = w2(z1, false);
  return softmax(z2);
};

/**
 * @brief Benchmark for the forward propagation through the neural network.
 *
 * This function reads data from a CSV file, initializes the layers, 
 * and measures the performance of the forward pass for a given number of samples.
 *
 * @param state Benchmark state object that manages the benchmarking process.
 */
void BM_EngineForwardProp(benchmark::State& state) {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  data.resize(state.range(0));
  label.resize(state.range(0));
  int n = static_cast<int>(data.size());

  int nin = 28 * 28;
  Layer w1(nin, 256);
  Layer w2(256, 10);

  for (auto _ : state) {
    std::vector<std::vector<std::shared_ptr<Value>>> y_pred;
    for (const std::vector<double>& item : data) {
      std::vector<std::shared_ptr<Value>> ptrs;
      ptrs.reserve(item.size());
      for (double i : item)
        ptrs.push_back(std::make_shared<Value>(Value(i)));
      y_pred.push_back(forward(ptrs, w1, w2));
    }
  }
}

BENCHMARK(BM_EngineForwardProp)->Arg(100);

/**
 * @brief Benchmark for the backward propagation through the neural network.
 *
 * This function reads data from a CSV file, initializes the layers, 
 * and measures the performance of the backward pass, calculating the 
 * loss using cross-entropy.
 *
 * @param state Benchmark state object that manages the benchmarking process.
 */
void BM_EngineBackwardProp(benchmark::State& state) {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  data.resize(state.range(0));
  label.resize(state.range(0));
  int n = static_cast<int>(data.size());

  int nin = 28 * 28;
  Layer w1(nin, 256);
  Layer w2(256, 10);

  std::vector<std::vector<std::shared_ptr<Value>>> y_pred;
  for (const std::vector<double>& item : data) {
    std::vector<std::shared_ptr<Value>> ptrs;
    ptrs.reserve(item.size());
    for (double i : item)
      ptrs.push_back(std::make_shared<Value>(Value(i)));
    y_pred.push_back(forward(ptrs, w1, w2));
  }

  for (auto _ : state) {
    std::shared_ptr<Value> loss = std::make_shared<Value>(Value(0));
    for (int i = 0; i < n; i++) {
      std::shared_ptr<Value> cross_entropy = std::make_shared<Value>(Value(0));
      for (int j = 0; j < 10; j++)
        if (j == label[i])
          cross_entropy = cross_entropy + y_pred[i][j]->log();
      loss = loss - cross_entropy;
    }
  }
}

// BENCHMARK(BM_EngineBackwardProp)->Arg(10);

}  // namespace

BENCHMARK_MAIN();