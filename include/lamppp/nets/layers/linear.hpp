#pragma once

#include "lamppp/autograd/variable.hpp"
#include "lamppp/nets/module.hpp"
#include "lamppp/nets/parameter.hpp"

namespace lmp::nets {

using ssize_t = int; // signed size_t

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

class FlattenImpl : public ModuleImpl {
 public:
  explicit FlattenImpl(ssize_t start_dim = 1, ssize_t end_dim = -1);
  autograd::Variable forward(const autograd::Variable& x) const;
 
 private:
   ssize_t start_dim_;
   ssize_t end_dim_;
};
LMP_DEFINE_MODULE(Flatten);

}