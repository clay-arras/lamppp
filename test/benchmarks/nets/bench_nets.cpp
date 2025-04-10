#include <benchmark/benchmark.h>
#include <vector>
#include "autograd/engine/variable.h"
#include "autograd/util/csv_reader.h"
#include "fast_layer.h"
#include "nn.h"

using std::vector;

namespace {

std::vector<Variable> softmax(
    std::vector<Variable> x) {
  assert((int)x.size() == 10);
  Variable denom = Variable(1e-4);
  for (const auto& i : x)
    denom = denom + i.exp();
  for (auto& i : x)
    i = i.exp() / denom;
  return x;
};

std::vector<Variable> forward(
    const std::vector<Variable>& x, Layer& w1, Layer& w2) {
  std::vector<Variable> z1 = w1(std::move(x));
  std::vector<Variable> z2 = w2(z1, false);
  return softmax(z2);
};

void BM_EngineForwardProp(benchmark::State& state) {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  data.resize(state.range(0));
  label.resize(state.range(0));
  int n = static_cast<int>(data.size());

  int nin = 28 * 28;
  Layer w1(nin, 256);
  Layer w2(256, 10);

  for (auto _ : state) {
    std::vector<std::vector<Variable>> y_pred;
    for (const std::vector<double>& item : data) {
      std::vector<Variable> ptrs;
      ptrs.reserve(item.size());
      for (double i : item)
        ptrs.emplace_back(i);
      y_pred.push_back(forward(ptrs, w1, w2));
    }
  }
}

BENCHMARK(BM_EngineForwardProp)->Arg(100);

void BM_EngineBackwardProp(benchmark::State& state) {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  data.resize(state.range(0));
  label.resize(state.range(0));
  int n = static_cast<int>(data.size());

  int nin = 28 * 28;
  Layer w1(nin, 256);
  Layer w2(256, 10);

  std::vector<std::vector<Variable>> y_pred;
  for (const std::vector<double>& item : data) {
    std::vector<Variable> ptrs;
    ptrs.reserve(item.size());
    for (double i : item)
      ptrs.emplace_back(i);
    y_pred.push_back(forward(ptrs, w1, w2));
  }

  for (auto _ : state) {
    Variable loss = Variable(0);
    for (int i = 0; i < n; i++) {
      Variable cross_entropy = Variable(0);
      for (int j = 0; j < 10; j++)
        if (j == label[i])
          cross_entropy = cross_entropy + y_pred[i][j].log();
      loss = loss - cross_entropy;
    }
  }
}

// BENCHMARK(BM_EngineBackwardProp)->Arg(10);

}  // namespace

namespace {

Variable relu_func(const Variable& v) {
    return Variable(v.relu());
}

Variable exp_func(const Variable& v) {
    return v.exp();
}

Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> relu(Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic>& x) {
    x = x.unaryExpr(&relu_func);
    return x;
}

Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> softmax(Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic>& x) {
    // for (auto row=x.rowwise().begin(); row != x.rowwise().end(); ++row) {
    //   *row = row->unaryExpr(&exp_func);
    //   Variable denom = Variable(1e-4) + row->sum();
    //   *row = *row / denom;
    // }
    return x;
}

Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> forward(
    Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic>& x, FastLayer& w1,
    FastLayer& w2) {
    Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> z1 = w1(x, relu);
    Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> z2 = w2(z1, softmax);
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

  Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> input_matrix(data.size(), 28 * 28);
  for (size_t i = 0; i < data.size(); ++i) {
      for (size_t j = 0; j < data[i].size(); ++j) {
          input_matrix(i, j) = Variable(data[i][j]);
      }
  }

  for (auto _ : state) {
    Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic>
      y_pred = forward(input_matrix, w1, w2);
  }
}

BENCHMARK(BM_FastEngineForwardProp)->Arg(100);

void BM_FastEngineBackwardProp(benchmark::State& state) {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  data.resize(state.range(0));
  label.resize(state.range(0));
  int n = static_cast<int>(data.size());

  int nin = 28 * 28;
  FastLayer w1(nin, 256);
  FastLayer w2(256, 10);

  std::vector<Eigen::Matrix<Variable, Eigen::Dynamic, 1>> y_pred;
  for (const std::vector<double>& item : data) {
    Eigen::Matrix<Variable, Eigen::Dynamic, 1> x(item.size(), 1);

    std::vector<Variable> shared_values;
    shared_values.reserve(item.size());
    for (double val : item) {
      shared_values.emplace_back(val);
    }
    x = Eigen::Map<Eigen::Matrix<Variable, Eigen::Dynamic, 1>>(
        shared_values.data(), shared_values.size());
    y_pred.push_back(forward(x, w1, w2));
  }

  for (auto _ : state) {
    Variable loss = Variable(0);
    for (int i = 0; i < n; i++) {
      Variable cross_entropy = Variable(0);
      for (int j = 0; j < 10; j++) {
        if (j == label[i]) {
          Variable pred_value = y_pred[i](j, 0) + Variable(1e-10);
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