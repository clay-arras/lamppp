#include <iostream>
#include <memory>
#include "autograd/engine/engine.h"
#include "autograd/nn/nn.h"

int main() {
  auto net =
      std::make_shared<MultiLayerPerceptron>(3, std::vector<int>{4, 4, 1});

  std::vector<std::vector<std::shared_ptr<Value>>> xs = {
      {std::make_shared<Value>(2.0), std::make_shared<Value>(3.0),
       std::make_shared<Value>(-1.0)},
      {std::make_shared<Value>(3.0), std::make_shared<Value>(-1.0),
       std::make_shared<Value>(0.5)},
      {std::make_shared<Value>(0.5), std::make_shared<Value>(1.0),
       std::make_shared<Value>(1.0)},
      {std::make_shared<Value>(1.0), std::make_shared<Value>(1.0),
       std::make_shared<Value>(-1.0)}};

  std::vector<double> ys = {1.0, -1.0, -1.0, 1.0};
  std::vector<std::vector<std::shared_ptr<Value>>> ypred;

  ypred.reserve(xs.size());
  for (const auto& x : xs) {
    ypred.push_back((*net)(x));
  }

  double alpha = 0.01;
  for (int i = 0; i < 100; i++) {
    std::vector<std::vector<std::shared_ptr<Value>>> ypred;

    ypred.reserve(xs.size());
    for (const auto& x : xs) {
      ypred.push_back((*net)(x));
    }

    auto loss = std::make_shared<Value>(0.0);
    for (size_t j = 0; j < ys.size(); j++) {
      auto diff = (ypred[j][0]) - std::make_shared<Value>(ys[j]);
      loss = loss + (diff->pow(std::make_shared<Value>(2.0)));
    }
    std::cout << "Loss: " << loss->data << std::endl;

    loss->backprop();
    auto params = net->parameters();
    for (auto& param : params) {
      param->data -= alpha * param->grad;
      param->grad = 0.0;
    }
  }

  return 0;
}