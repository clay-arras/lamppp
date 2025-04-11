#ifndef _NN_H_
#define _NN_H_

#include <memory>
#include <vector>
#include "variable.h"

class Neuron : public std::enable_shared_from_this<Neuron> {
 private:
  std::vector<Variable> weights_;  ///< Weights of the neuron.
  Variable bias_;                  ///< Bias of the neuron.

 public:
  explicit Neuron(int nin);
  std::vector<Variable> parameters();
  Variable operator()(
      const std::vector<Variable>& x);
};

class Layer : public std::enable_shared_from_this<Layer> {
 private:
  std::vector<std::shared_ptr<Neuron>> neurons_;  ///< Neurons in the layer.

 public:
  Layer(int nin, int nout);
  std::vector<Variable> parameters();
  std::vector<Variable> operator()(
      const std::vector<Variable>& x, bool activ = true);
};

class MultiLayerPerceptron
    : public std::enable_shared_from_this<MultiLayerPerceptron> {
 private:
  std::vector<std::shared_ptr<Layer>> layers_;  ///< Layers in the MLP.

 public:
  MultiLayerPerceptron(int nin, const std::vector<int>& nouts);
  std::vector<Variable> parameters();
  std::vector<Variable> operator()(
      const std::vector<Variable>& x);
};

#endif  //_NN_H_