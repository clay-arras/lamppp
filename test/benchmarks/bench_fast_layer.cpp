#include <benchmark/benchmark.h>
#include <vector>
#include "autograd/engine/wrapper_engine.h"
#include "autograd/nn/fast_layer.h"
#include "autograd/util/csv_reader.h"

using std::vector;

namespace { 

/**
 * @brief Applies the ReLU activation function to a given input matrix.
 *
 * This function modifies the input matrix in place by applying the ReLU function
 * to each element, setting negative values to zero.
 *
 * @param x Input matrix of SharedValue elements.
 * @return Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> The modified input matrix after applying ReLU.
 */
std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic,
1>(Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>&)> relu =
[](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>& x) {
    auto relu_func = [](const SharedValue& v) -> SharedValue {
        return SharedValue(v.getPtr()->relu());
    };
    x = x.unaryExpr(relu_func);
    return x;
};

/**
 * @brief Applies the softmax function to a given input matrix.
 *
 * This function computes the softmax of the input matrix, which is a common
 * activation function used in multi-class classification problems. It normalizes
 * the input values to a probability distribution.
 *
 * @param x Input matrix of SharedValue elements.
 * @return Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> The modified input matrix after applying softmax.
 */
std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic,
1>(Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>&)> softmax =
[](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>& x) {
    assert(x.rows() == 10);
    x = x.unaryExpr([](const SharedValue& v) { return v.exp(); });
    SharedValue denom = SharedValue(1e-4) + x.sum();
    x = x / denom;
    return x;
};

/**
 * @brief Performs a forward pass through the two FastLayers.
 *
 * This function takes an input matrix and applies the first layer followed by
 * the second layer, using the ReLU and softmax activation functions respectively.
 *
 * @param x Input matrix of SharedValue elements.
 * @param w1 First FastLayer to apply.
 * @param w2 Second FastLayer to apply.
 * @return Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> The output matrix after applying both layers.
 */
Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> forward(
    Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>& x,
    FastLayer& w1, 
    FastLayer& w2) {
  Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> z1 = w1(x, relu);
  Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> z2 = w2(z1, softmax);
  return z2;
}

/**
 * @brief Benchmark for the forward propagation through the FastLayer.
 *
 * This function reads data from a CSV file, initializes the FastLayers, and
 * measures the performance of the forward pass for a given number of samples.
 *
 * @param state Benchmark state object that manages the benchmarking process.
 */
void BM_FastEngineForwardProp(benchmark::State& state) {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  data.resize(state.range(0));
  label.resize(state.range(0));
  int n = static_cast<int>(data.size());

  int nin = 28 * 28;
  FastLayer w1(nin, 256);
  FastLayer w2(256, 10);

  for (auto _ : state) {
    std::vector<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>> y_pred;
    for (const std::vector<double>& item : data) {
      Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x(item.size(), 1);

      std::vector<SharedValue> shared_values;
      shared_values.reserve(item.size());
      for (double val : item) {
        shared_values.emplace_back(val);
      }
      x = Eigen::Map<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>>(
          shared_values.data(), shared_values.size());
      y_pred.push_back(forward(x, w1, w2));
    }
  }
}

BENCHMARK(BM_FastEngineForwardProp)->Arg(100)->Arg(1000);

/**
 * @brief Benchmark for the backward propagation through the FastLayer.
 *
 * This function reads data from a CSV file, initializes the FastLayers, and
 * measures the performance of the backward pass, calculating the loss using
 * cross-entropy.
 *
 * @param state Benchmark state object that manages the benchmarking process.
 */
void BM_FastEngineBackwardProp(benchmark::State& state) {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  data.resize(state.range(0));
  label.resize(state.range(0));
  int n = static_cast<int>(data.size());

  int nin = 28 * 28;
  FastLayer w1(nin, 256);
  FastLayer w2(256, 10);

  std::vector<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>> y_pred;
  for (const std::vector<double>& item : data) {
    Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x(item.size(), 1);

    std::vector<SharedValue> shared_values;
    shared_values.reserve(item.size());
    for (double val : item) {
      shared_values.emplace_back(val);
    }
    x = Eigen::Map<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>>(
        shared_values.data(), shared_values.size());
    y_pred.push_back(forward(x, w1, w2));
  }

  for (auto _ : state) {
    SharedValue loss = SharedValue(0);
    for (int i = 0; i < n; i++) {
        SharedValue cross_entropy = SharedValue(0);
        for (int j = 0; j < 10; j++) {
            if (j == label[i]) {
                SharedValue pred_value = y_pred[i](j, 0) + SharedValue(1e-10);
                cross_entropy = cross_entropy + pred_value.log();
            }
        }
        loss = loss - cross_entropy;
    }
  }
}

// BENCHMARK(BM_FastEngineBackwardProp)->Arg(10);

}  // namespace

BENCHMARK_MAIN();