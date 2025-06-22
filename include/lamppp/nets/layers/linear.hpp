#pragma once

#include "lamppp/autograd/constructor.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/nets/module.hpp"
#include "lamppp/nets/parameter.hpp"

namespace lmp::nets {

class LinearImpl : public ModuleImpl {
 public:
  explicit LinearImpl(size_t in_features, size_t out_features, bool bias = true,
             tensor::DeviceType device = DEFAULT_DEVICE,
             tensor::DataType dtype = DEFAULT_DTYPE);
  autograd::Variable forward(const autograd::Variable& x) const;

 private:
  Parameter weights_;  
  Parameter bias_;     
  bool requires_bias_;
};
LMP_DEFINE_MODULE(Linear);

}