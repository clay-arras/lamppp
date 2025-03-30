#ifndef _FAST_LAYER_H_
#define _FAST_LAYER_H_

#include <Eigen/Core>
#include <functional>
#include <initializer_list>
#include <memory>
#include <random>
#include "engine.h"
#include "wrapper_engine.h"

class FastLayer {
 private:
  int nin, nout;
  Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> weights;
  Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> bias;

 public:
  FastLayer(int nin, int nout);

  Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> operator()(
      Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>& x,
      std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>(
          Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>&)>
          activ);
};

#endif  // _FAST_LAYER_H_
