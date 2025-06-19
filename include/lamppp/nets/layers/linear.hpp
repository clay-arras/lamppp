#pragma once

#include "lamppp/autograd/constructor.hpp"
#include "lamppp/autograd/variable.hpp"
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
  autograd::Variable forward(const autograd::Variable& x) const;

 private:
  Parameter weights_;  
  Parameter bias_;     
  bool requires_bias_;
};

struct Linear : public ModuleCRTP<LinearImpl> {
    Linear(size_t in_features, size_t out_features, bool bias = true, tensor::DeviceType device = DEFAULT_DEVICE,
                  tensor::DataType dtype = DEFAULT_DTYPE);
};

}