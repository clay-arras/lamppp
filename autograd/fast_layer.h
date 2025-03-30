#ifndef _FAST_LAYER_H_
#define _FAST_LAYER_H_

#include "engine.h"
#include <Eigen/Core>
#include <functional>
#include <initializer_list>
#include <memory>
#include <random>
#include "wrapper_engine.h"

class FastLayer {
private:
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> weights;
    Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> bias;

public:
    FastLayer(int nin, int nout);
    
    Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>
    operator()(Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x, 
               std::function<SharedValue(SharedValue)> activ);
};

#endif // _FAST_LAYER_H_
