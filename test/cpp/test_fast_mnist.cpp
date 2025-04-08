#include <memory>
#include <vector>
#include "autograd/engine/engine.h"
#include "autograd/nn/nn.h"
#include "autograd/util/csv_reader.h"

using std::vector;

namespace {

std::vector<std::shared_ptr<Value>> softmax(
    std::vector<std::shared_ptr<Value>> x) {
  assert((int)x.size() == 10);
  std::shared_ptr<Value> denom = std::make_shared<Value>(Value(1e-4));
  for (const auto& i : x)
    denom = denom + i->exp();
  for (auto& i : x)
    i = i->exp() / denom;
  return x;
}

std::vector<std::shared_ptr<Value>> forward(
    const std::vector<std::shared_ptr<Value>>& x, Layer& w1, Layer& w2) {
  std::vector<std::shared_ptr<Value>> z1 = w1(std::move(x));
  std::vector<std::shared_ptr<Value>> z2 = w2(z1, false);
  return softmax(z2);
}

}

int main() {
  auto [data, label] = readCSV("data/mnist_dummy.csv");
  data.resize(1000);
  label.resize(1000);
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

  return 0;
}
