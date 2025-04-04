#include "nn.h"
#include "autograd/engine/engine.h"
#include <random>
#include <utility>

/**
 * @brief Constructs a Neuron with the specified number of input connections.
 *
 * This constructor initializes the weights and bias of the neuron with random values
 * drawn from a uniform distribution in the range [-1.0, 1.0].
 *
 * @param nin Number of input connections to the neuron.
 */
Neuron::Neuron(int nin) {
  std::random_device seed;
  std::mt19937 gen{seed()};
  std::uniform_real_distribution<> dist{-1.0, 1.0};
  for (int i = 0; i < nin; i++) {
    this->weights_.push_back(std::make_shared<Value>(dist(gen)));
  }
  this->bias_ = std::make_shared<Value>(dist(gen));
}

/**
 * @brief Retrieves the parameters of the neuron.
 *
 * This function returns a vector containing the weights and bias of the neuron.
 *
 * @return A vector of shared pointers to the Value objects representing the weights and bias.
 */
std::vector<std::shared_ptr<Value>> Neuron::parameters() {
  std::vector<std::shared_ptr<Value>> ret(this->weights_.begin(),
                                          this->weights_.end());
  ret.push_back(this->bias_);
  return ret;
}

/**
 * @brief Computes the output of the neuron given an input vector.
 *
 * This operator takes an input vector, applies the neuron's weights and bias,
 * and returns the resulting output. The output is computed as the weighted sum
 * of the inputs plus the bias.
 *
 * @param x A vector of shared pointers to the input values.
 * @return A shared pointer to the output value after applying the weights and bias.
 */
std::shared_ptr<Value> Neuron::operator()(
    const std::vector<std::shared_ptr<Value>> &x) {
  std::shared_ptr<Value> ret = std::make_shared<Value>(this->bias_->data);
  for (int i = 0; i < static_cast<int>(x.size()); i++) {
    ret = ret + (this->weights_[i] * x[i]);
  }
  return ret;
}

/**
 * @brief Constructs a Layer with the specified number of input and output neurons.
 *
 * This constructor initializes the layer by creating a specified number of neurons,
 * each connected to the inputs of the layer.
 *
 * @param nin Number of input features to the layer.
 * @param nout Number of output neurons in the layer.
 */
Layer::Layer(int nin, int nout) {
  for (int i = 0; i < nout; i++) {
    this->neurons_.push_back(std::make_shared<Neuron>(nin));
  }
}

/**
 * @brief Retrieves the parameters of the layer.
 *
 * This function returns a vector containing the parameters (weights and biases)
 * of all neurons in the layer.
 *
 * @return A vector of shared pointers to the Value objects representing the parameters of the layer.
 */
std::vector<std::shared_ptr<Value>> Layer::parameters() {
  std::vector<std::shared_ptr<Value>> params;
  for (auto & neuron : this->neurons_) {
    std::vector<std::shared_ptr<Value>> n_params =
        neuron->parameters();
    params.insert(params.end(), n_params.begin(), n_params.end());
  }
  return params;
}

/**
 * @brief Computes the output of the layer given an input vector.
 *
 * This operator takes an input vector, processes it through each neuron in the layer,
 * and applies an activation function if specified.
 *
 * @param x A vector of shared pointers to the input values.
 * @param activ A boolean indicating whether to apply the activation function.
 * @return A vector of shared pointers to the output values after processing through the layer.
 */
std::vector<std::shared_ptr<Value>> Layer::operator()(
    const std::vector<std::shared_ptr<Value>> &x, bool activ) {
  std::vector<std::shared_ptr<Value>> ret;
  for (auto & neuron : this->neurons_) {
    std::shared_ptr<Value> val = (*neuron)(x);
    if (activ) {
      ret.push_back(val->relu());
    } else {
      ret.push_back(val);
    }
  }
  return ret;
}

/**
 * @brief Constructs a MultiLayerPerceptron with the specified input and output sizes.
 *
 * This constructor initializes the multi-layer perceptron by creating layers
 * based on the specified number of input features and output sizes for each layer.
 *
 * @param nin Number of input features to the multi-layer perceptron.
 * @param nouts A vector containing the number of output neurons for each layer.
 */
MultiLayerPerceptron::MultiLayerPerceptron(int nin, const std::vector<int> &nouts) {
  int prev = nin;
  for (int nout : nouts) {
    this->layers_.push_back(std::make_shared<Layer>(prev, nout));
    prev = nout;
  }
}

/**
 * @brief Retrieves the parameters of the multi-layer perceptron.
 *
 * This function returns a vector containing the parameters (weights and biases)
 * of all layers in the multi-layer perceptron.
 *
 * @return A vector of shared pointers to the Value objects representing the parameters of the MLP.
 */
std::vector<std::shared_ptr<Value>> MultiLayerPerceptron::parameters() {
  std::vector<std::shared_ptr<Value>> params;
  for (auto & layer : this->layers_) {
    std::vector<std::shared_ptr<Value>> l_params =
        layer->parameters();
    params.insert(params.end(), l_params.begin(), l_params.end());
  }
  return params;
}

/**
 * @brief Computes the output of the multi-layer perceptron given an input vector.
 *
 * This operator takes an input vector and processes it through each layer of the
 * multi-layer perceptron, returning the final output.
 *
 * @param x A vector of shared pointers to the input values.
 * @return A vector of shared pointers to the output values after processing through the MLP.
 */
std::vector<std::shared_ptr<Value>> MultiLayerPerceptron::operator()(
    const std::vector<std::shared_ptr<Value>> &x) {
  std::vector<std::shared_ptr<Value>> ret = std::move(x);
  for (auto & layer : this->layers_) {
    ret = (*layer)(ret);
  }
  return ret;
}