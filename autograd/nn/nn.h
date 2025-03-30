#ifndef _NN_H_
#define _NN_H_

#include <memory>
#include <vector>
#include "autograd/engine/engine.h"

/**
 * @brief Represents a single neuron in a neural network.
 *
 * This class encapsulates the weights and bias for a neuron, and provides
 * functionality to compute the output given an input vector.
 */
class Neuron : public std::enable_shared_from_this<Neuron> {
 private:
  std::vector<std::shared_ptr<Value>> weights_; ///< Weights of the neuron.
  std::shared_ptr<Value> bias_;                 ///< Bias of the neuron.

 public:
  /**
   * @brief Constructs a Neuron with the specified number of input connections.
   *
   * This constructor initializes the weights and bias of the neuron with random values
   * drawn from a uniform distribution in the range [-1.0, 1.0].
   *
   * @param nin Number of input connections to the neuron.
   */
  explicit Neuron(int nin);

  /**
   * @brief Retrieves the parameters of the neuron.
   *
   * This function returns a vector containing the weights and bias of the neuron.
   *
   * @return A vector of shared pointers to the Value objects representing the weights and bias.
   */
  std::vector<std::shared_ptr<Value>> parameters();

  /**
   * @brief Computes the output of the neuron given an input vector.
   *
   * This operator takes an input vector, applies the neuron's weights and bias,
   * and returns the resulting output.
   *
   * @param x A vector of shared pointers to the input values.
   * @return A shared pointer to the output value after applying the weights and bias.
   */
  std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>> x);
};

/**
 * @brief Represents a layer of neurons in a neural network.
 *
 * This class encapsulates multiple neurons and provides functionality to process
 * input through the layer.
 */
class Layer : public std::enable_shared_from_this<Layer> {
 private:
  std::vector<std::shared_ptr<Neuron>> neurons_; ///< Neurons in the layer.

 public:
  /**
   * @brief Constructs a Layer with the specified number of input and output neurons.
   *
   * This constructor initializes the layer by creating a specified number of neurons,
   * each connected to the inputs of the layer.
   *
   * @param nin Number of input features to the layer.
   * @param nout Number of output neurons in the layer.
   */
  Layer(int nin, int nout);

  /**
   * @brief Retrieves the parameters of the layer.
   *
   * This function returns a vector containing the parameters (weights and biases)
   * of all neurons in the layer.
   *
   * @return A vector of shared pointers to the Value objects representing the parameters of the layer.
   */
  std::vector<std::shared_ptr<Value>> parameters();

  /**
   * @brief Computes the output of the layer given an input vector.
   *
   * This operator takes an input vector and processes it through each neuron in the layer,
   * returning the final output.
   *
   * @param x A vector of shared pointers to the input values.
   * @param activ A boolean indicating whether to apply the activation function.
   * @return A vector of shared pointers to the output values after processing through the layer.
   */
  std::vector<std::shared_ptr<Value>> operator()(
      std::vector<std::shared_ptr<Value>> x, bool activ = true);
};

/**
 * @brief Represents a multi-layer perceptron (MLP) neural network.
 *
 * This class encapsulates multiple layers and provides functionality to process
 * input through the entire network.
 */
class MultiLayerPerceptron
    : public std::enable_shared_from_this<MultiLayerPerceptron> {
 private:
  std::vector<std::shared_ptr<Layer>> layers_; ///< Layers in the MLP.

 public:
  /**
   * @brief Constructs a MultiLayerPerceptron with the specified input and output sizes.
   *
   * This constructor initializes the multi-layer perceptron by creating layers
   * based on the specified number of input features and output sizes for each layer.
   *
   * @param nin Number of input features to the multi-layer perceptron.
   * @param nouts A vector containing the number of output neurons for each layer.
   */
  MultiLayerPerceptron(int nin, std::vector<int> nouts);

  /**
   * @brief Retrieves the parameters of the multi-layer perceptron.
   *
   * This function returns a vector containing the parameters (weights and biases)
   * of all layers in the multi-layer perceptron.
   *
   * @return A vector of shared pointers to the Value objects representing the parameters of the MLP.
   */
  std::vector<std::shared_ptr<Value>> parameters();

  /**
   * @brief Computes the output of the multi-layer perceptron given an input vector.
   *
   * This operator takes an input vector and processes it through each layer of the
   * multi-layer perceptron, returning the final output.
   *
   * @param x A vector of shared pointers to the input values.
   * @return A vector of shared pointers to the output values after processing through the MLP.
   */
  std::vector<std::shared_ptr<Value>> operator()(
      std::vector<std::shared_ptr<Value>> x);
};

#endif //_NN_H_