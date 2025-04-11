#include "nn.h"
#include <random>
#include <utility>

Neuron::Neuron(int nin) {
  std::random_device seed;
  std::mt19937 gen{seed()};
  std::uniform_real_distribution<> dist{-1.0, 1.0};
  for (int i = 0; i < nin; i++) {
    this->weights_.emplace_back(dist(gen));
  }
  this->bias_ = Variable(dist(gen));
}

std::vector<Variable> Neuron::parameters() {
  std::vector<Variable> ret(this->weights_.begin(), this->weights_.end());
  ret.push_back(this->bias_);
  return ret;
}

Variable Neuron::operator()(const std::vector<Variable>& x) {
  Variable ret = Variable(this->bias_.data());
  for (int i = 0; i < static_cast<int>(x.size()); i++) {
    ret = ret + (this->weights_[i] * x[i]);
  }
  return ret;
}

Layer::Layer(int nin, int nout) {
  for (int i = 0; i < nout; i++) {
    this->neurons_.push_back(std::make_shared<Neuron>(nin));
  }
}

std::vector<Variable> Layer::parameters() {
  std::vector<Variable> params;
  for (auto& neuron : this->neurons_) {
    std::vector<Variable> n_params = neuron->parameters();
    params.insert(params.end(), n_params.begin(), n_params.end());
  }
  return params;
}

std::vector<Variable> Layer::operator()(const std::vector<Variable>& x,
                                        bool activ) {
  std::vector<Variable> ret;
  for (auto& neuron : this->neurons_) {
    Variable val = (*neuron)(x);
    if (activ) {
      ret.push_back(val.relu());
    } else {
      ret.push_back(val);
    }
  }
  return ret;
}

MultiLayerPerceptron::MultiLayerPerceptron(int nin,
                                           const std::vector<int>& nouts) {
  int prev = nin;
  for (int nout : nouts) {
    this->layers_.push_back(std::make_shared<Layer>(prev, nout));
    prev = nout;
  }
}

std::vector<Variable> MultiLayerPerceptron::parameters() {
  std::vector<Variable> params;
  for (auto& layer : this->layers_) {
    std::vector<Variable> l_params = layer->parameters();
    params.insert(params.end(), l_params.begin(), l_params.end());
  }
  return params;
}

std::vector<Variable> MultiLayerPerceptron::operator()(
    const std::vector<Variable>& x) {
  std::vector<Variable> ret = std::move(x);
  for (auto& layer : this->layers_) {
    ret = (*layer)(ret);
  }
  return ret;
}