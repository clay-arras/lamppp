#ifndef _FAST_LAYER_H_
#define _FAST_LAYER_H_

#include "engine.h"
#include <Eigen/Core>
#include <functional>
#include <initializer_list>
#include <memory>
#include <random>

class FastLayer {
private:
    Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, Eigen::Dynamic> weights;
    Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1> bias;

public:
    FastLayer(int nin, int nout);
    
    Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1>
    operator()(Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1> x, 
               std::function<std::shared_ptr<Value>(std::shared_ptr<Value>)> activ);
};

#endif // _FAST_LAYER_H_
