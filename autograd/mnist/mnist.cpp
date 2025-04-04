#include "mnist.h"

#include <cassert>
#include <memory>
#include <utility>
#include <vector>
#include "autograd/engine/engine.h"
#include "autograd/nn/nn.h"
#include "autograd/util/csv_reader.h"

int main() {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  int n = static_cast<int>(data.size());

  int nin = 28 * 28;
  Layer w1(nin, 256);
  Layer w2(256, 10);

  auto softmax = [&](std::vector<std::shared_ptr<Value>> x)
      -> std::vector<std::shared_ptr<Value>> {
    assert((int)x.size() == 10);
    std::shared_ptr<Value> denom = std::make_shared<Value>(Value(1e-4));
    for (const auto& i : x)
      denom = denom + i->exp();
    for (auto& i : x)
      i = i->exp() / denom;
    return x;
  };

  auto forward = [&](const std::vector<std::shared_ptr<Value>>& x) {
    std::vector<std::shared_ptr<Value>> z1 = w1(std::move(x));
    std::vector<std::shared_ptr<Value>> z2 = w2(z1, false);
    return softmax(z2);
  };

  std::vector<std::vector<std::shared_ptr<Value>>> y_pred;
  for (const std::vector<double>& item : data) {
    std::vector<std::shared_ptr<Value>> ptrs;
    ptrs.reserve(item.size());
    for (double i : item)
      ptrs.push_back(std::make_shared<Value>(Value(i)));
    y_pred.push_back(forward(ptrs));
  }
  // std::shared_ptr<Value> loss = std::make_shared<Value>(Value(0));

  // for (int i=0; i<N; i++) {
  //     std::shared_ptr<Value> cross_entropy =
  //     std::make_shared<Value>(Value(0)); for (int j=0; j<10; j++)
  //         if (j == label[i])
  //             cross_entropy = cross_entropy + y_pred[i][j]->log();
  //     loss = loss - cross_entropy;
  // }
  // std::cout << loss->data << std::endl;
}