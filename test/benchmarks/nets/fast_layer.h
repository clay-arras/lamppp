#ifndef _FAST_LAYER_H_
#define _FAST_LAYER_H_

#include <Eigen/Core>
#include "autograd/engine/variable.h"

class FastLayer {
 private:
  int nin_;
  int nout_;
  Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> weights_;
  Eigen::Matrix<Variable, 1, Eigen::Dynamic> bias_;

 public:
  FastLayer(int nin, int nout);

  Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> operator()(
      Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic>& x,
      const std::function<Eigen::Matrix<Variable, Eigen::Dynamic,
                                        Eigen::Dynamic>(
          Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic>&)>& activ);
};

#endif  // _FAST_LAYER_H_
