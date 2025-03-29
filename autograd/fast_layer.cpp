#include "layer.h"
#include "Eigen/src/Core/util/Constants.h"
#include "engine.h"
#include <Eigen/Core>
#include <functional>
#include <initializer_list>
#include <memory>

Neuron::Neuron(int nin) {
  std::random_device seed;
  std::mt19937 gen{seed()};
  std::uniform_real_distribution<> dist{-1.0, 1.0};
  for (int i = 0; i < nin; i++) {
    this->weights.push_back(std::make_shared<Value>(dist(gen)));
  }
  this->bias = std::make_shared<Value>(dist(gen));
}

std::shared_ptr<Value>
Neuron::operator()(std::vector<std::shared_ptr<Value>> x) {
  std::shared_ptr<Value> ret = std::make_shared<Value>(this->bias->data);
  for (int i = 0; i < (int)x.size(); i++) {
    ret = ret + (this->weights[i] * x[i]);
  }
  return ret;
}

FastLayer::FastLayer(int nin, int nout) { 
    Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, Eigen::Dynamic> W;
    Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1> b;
    W.resize(nout, nin);
    b.resize(nout, 1);

    std::random_device seed;
    std::mt19937 gen{seed()};
    std::uniform_real_distribution<> dist{-1.0, 1.0};
    auto initValue = [](const std::shared_ptr<Value> &a) {
        return std::make_shared<Value>(dist(gen));
    };
    W = W.unaryExpr(initValue);
    b = b.unaryExpr(initValue);
}

Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1>
FastLayer::operator()(Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1> x, std::function<std::shared_ptr<Value>(std::shared_ptr<Value>)> activ) {
    Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1> z = bias;

    // multiply the two matrices
    z = z.unaryExpr(activ);
    return z;
}