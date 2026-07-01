#pragma once

#include "lamp3/autograd/variable.hpp"
#include "lamp3/nets/module.hpp"
#include "lamp3/tensor/data_type.hpp"

namespace lmp::nets {

class DropoutImpl : public ModuleImpl {
 public:
  explicit DropoutImpl(tensor::Scalar p) : p_(p) {};
  autograd::Variable forward(const autograd::Variable& x) const;

 private:
    tensor::Scalar p_;
};
LMP_DEFINE_MODULE(Dropout);

}