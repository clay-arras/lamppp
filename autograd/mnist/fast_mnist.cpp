#include "fast_mnist.h"
#include <cassert>
#include <vector>
#include "autograd/engine/wrapper_engine.h"
#include "autograd/nn/fast_layer.h"
#include "autograd/util/csv_reader.h"

int main() {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  int n = static_cast<int>(data.size());

  int nin = 28 * 28;
  FastLayer w1(nin, 256);
  FastLayer w2(256, 10);

  std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>(
      Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>&)>
      relu = [](Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>& x) {
        auto relu_func = [](const SharedValue& v) -> SharedValue {
          return SharedValue(v.getPtr()->relu());
        };
        x = x.unaryExpr(relu_func);
        return x;
      };

  std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>(
      Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>&)>
      softmax = [&](Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>& x) {
        x = x.unaryExpr([](const SharedValue& v) { return v.exp(); });
        for (auto row = x.rowwise().begin(); row != x.rowwise().end(); ++row) {
          SharedValue denom = SharedValue(1e-4) + row->sum();
          *row = *row / denom;
        }
        return x;
      };

  auto forward = [&](Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>& x) {
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> z1 = w1(x, relu);
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> z2 = w2(z1, softmax);
    return z2;
  };

  std::vector<Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>> y_pred;
  for (const std::vector<double>& item : data) {
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> x(1, item.size());

    for (size_t j = 0; j < item.size(); ++j) {
      x(0, j) = SharedValue(item[j]);
    }
    
    y_pred.push_back(forward(x));
  }
  SharedValue loss = SharedValue(0);

  for (int i = 0; i < n; i++) {
    SharedValue cross_entropy = SharedValue(0);
    for (int j = 0; j < 10; j++) {
      if (j == label[i]) {
        SharedValue pred_value = y_pred[i](0, j) + SharedValue(1e-10);
        cross_entropy = cross_entropy + pred_value.log();
      }
    }
    loss = loss - cross_entropy;
  }
}