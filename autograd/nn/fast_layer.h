#ifndef _FAST_LAYER_H_
#define _FAST_LAYER_H_

#include <Eigen/Core>
#include "autograd/engine/wrapper_engine.h"

/**
 * @brief Represents a fast layer in a neural network.
 *
 * This class encapsulates the weights and bias for a layer, and provides
 * functionality to perform a forward pass through the layer using a specified
 * activation function.
 */
class FastLayer {
 private:
  int nin_;  ///< Number of input features.
  int nout_; ///< Number of output features.
  Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> weights_; ///< Weights of the layer.
  Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> bias_;               ///< Bias of the layer.

 public:
  /**
   * @brief Constructs a FastLayer with the specified number of input and output features.
   *
   * This constructor initializes the weights and bias of the layer with random values
   * drawn from a uniform distribution in the range [-1.0, 1.0].
   *
   * @param nin Number of input features.
   * @param nout Number of output features.
   */
  FastLayer(int nin, int nout);

  /**
   * @brief Forward pass through the FastLayer.
   *
   * This operator takes an input vector and applies the layer's weights and bias,
   * followed by an activation function.
   *
   * @param x Input vector of shape (nin x 1), where nin is the number of input
   * features.
   * @param activ Activation function to be applied to the output after the linear
   * transformation.
   *
   * @return Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> The output vector after
   * applying the weights, bias, and activation function, of shape (nout x 1),
   * where nout is the number of output features.
   */
  Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> operator()(
      Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>& x,
      const std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>(
          Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>&)>&
          activ);
};

#endif  // _FAST_LAYER_H_
