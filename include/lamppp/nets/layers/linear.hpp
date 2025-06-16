#pragma once

#include "lamppp/autograd/constructor.hpp"
#include "lamppp/autograd/functions/matrix_ops.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/autograd/core.hpp"
#include "lamppp/nets/module.hpp"
#include "lamppp/nets/parameter.hpp"

namespace lmp::nets {

class LinearImpl : public ModuleImpl {
 public:
  LinearImpl(size_t in_features, size_t out_features, bool bias = true,
             tensor::DeviceType device = DEFAULT_DEVICE,
             tensor::DataType dtype = DEFAULT_DTYPE)
      : requires_bias_(bias),
        weights_(autograd::randn(0, 1, {in_features, out_features}, true,
                                 device, dtype)),
        bias_(autograd::randn(0, 1, {out_features}, true, device, dtype)) {};

  autograd::Variable forward(const autograd::Variable& x) {
    if (!requires_bias_) {
        return autograd::ops::matmul(x, weights_);
    }
    return autograd::ops::matmul(x, weights_) + static_cast<autograd::Variable>(bias_);
  }

 private:
  Parameter weights_;  
  Parameter bias_;     
  bool requires_bias_;
};

class Linear : public Module<LinearImpl> {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true, tensor::DeviceType device = DEFAULT_DEVICE,
                  tensor::DataType dtype = DEFAULT_DTYPE) {
            impl_ = std::make_shared<LinearImpl>(LinearImpl(in_features, out_features, bias, device, dtype));
        };
};

}