#include <benchmark/benchmark.h>
#include "autograd/engine/wrapper_engine.h"
#include "autograd/nn/fast_layer.h"
#include "autograd/util/csv_reader.h"

namespace {

SharedValue relu_func(const SharedValue& v) {
    return SharedValue(v.getPtr()->relu());
}

SharedValue exp_func(const SharedValue& v) {
    return v.exp();
}

Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> relu(Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>& x) {
    x = x.unaryExpr(&relu_func);
    return x;
}

Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> softmax(Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>& x) {
    // for (auto row=x.rowwise().begin(); row != x.rowwise().end(); ++row) {
    //   *row = row->unaryExpr(&exp_func);
    //   SharedValue denom = SharedValue(1e-4) + row->sum();
    //   *row = *row / denom;
    // }
    return x;
}

Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> forward(
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>& x, FastLayer& w1,
    FastLayer& w2) {
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> z1 = w1(x, relu);
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> z2 = w2(z1, softmax);
    return z2;
}

void BM_FastEngineForwardProp(benchmark::State& state) {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  data.resize(state.range(0));
  label.resize(state.range(0));
  int n = static_cast<int>(data.size());

  int nin = 28 * 28;
  FastLayer w1(nin, 256);
  FastLayer w2(256, 10);

  Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> input_matrix(data.size(), 28 * 28);
  for (size_t i = 0; i < data.size(); ++i) {
      for (size_t j = 0; j < data[i].size(); ++j) {
          input_matrix(i, j) = SharedValue(data[i][j]);
      }
  }

  for (auto _ : state) {
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>
      y_pred = forward(input_matrix, w1, w2);
  }
}

BENCHMARK(BM_FastEngineForwardProp)->Arg(100);

void BM_FastEngineBackwardProp(benchmark::State& state) {
  // auto [data, label] = readCSV("data/mnist_dummy.csv");
  // data.resize(state.range(0));
  // label.resize(state.range(0));
  // int n = static_cast<int>(data.size());

  // int nin = 28 * 28;
  // FastLayer w1(nin, 256);
  // FastLayer w2(256, 10);

  // std::vector<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>> y_pred;
  // for (const std::vector<double>& item : data) {
  //   Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x(item.size(), 1);

  //   std::vector<SharedValue> shared_values;
  //   shared_values.reserve(item.size());
  //   for (double val : item) {
  //     shared_values.emplace_back(val);
  //   }
  //   x = Eigen::Map<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>>(
  //       shared_values.data(), shared_values.size());
  //   y_pred.push_back(forward(x, w1, w2));
  // }

  // for (auto _ : state) {
  //   SharedValue loss = SharedValue(0);
  //   for (int i = 0; i < n; i++) {
  //     SharedValue cross_entropy = SharedValue(0);
  //     for (int j = 0; j < 10; j++) {
  //       if (j == label[i]) {
  //         SharedValue pred_value = y_pred[i](j, 0) + SharedValue(1e-10);
  //         cross_entropy = cross_entropy + pred_value.log();
  //       }
  //     }
  //     loss = loss - cross_entropy;
  //   }
  // }
}

// BENCHMARK(BM_FastEngineBackwardProp)->Arg(10);

}  // namespace

BENCHMARK_MAIN();