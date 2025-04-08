#include "fast_layer.h"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <functional>
#include <random>
#include "Eigen/src/Core/util/Constants.h"

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
  weights_.resize(nin, nout);
  bias_.resize(1, nout);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  auto random_shared = [&dis, &gen](const SharedValue&) -> SharedValue {
    return SharedValue(dis(gen));
  };

  weights_ = weights_.unaryExpr(random_shared);
  bias_ = bias_.unaryExpr(random_shared);
}

Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> FastLayer::operator()(
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>& x,
    const std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>(
        Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>&)>& activ) {
  Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> a;
  a = x * weights_;
  // for (auto && row : x.rowwise()) {
  //   row = row + bias_;
  // }
  return activ(a);
}