#include "fast_layer.h"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <functional>
#include <random>

/**
 * @brief Constructs a FastLayer with the specified number of input and output features.
 *
 * This constructor initializes the weights and bias of the layer with random values
 * drawn from a uniform distribution in the range [-1.0, 1.0].
 *
 * @param nin Number of input features.
 * @param nout Number of output features.
 */
using Eigen::Matrix;

FastLayer::FastLayer(int nin, int nout) : nin_(nin), nout_(nout) {
  weights_.resize(nout, nin);
  bias_.resize(nout, 1);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  auto random_shared = [&dis, &gen](const SharedValue&) -> SharedValue {
      return SharedValue(dis(gen));
  };

  weights_ = weights_.unaryExpr(random_shared);
  bias_ = bias_.unaryExpr(random_shared);
}

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
Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> FastLayer::operator()(
    Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>& x,
    const std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>(
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>&)>& activ) {
  Matrix<SharedValue, Eigen::Dynamic, 1> a(nout_, 1);
  a = weights_ * x + bias_;
  return activ(a);
}