#pragma once

#include "lamppp/autograd/variable.hpp"
#include "lamppp/nets/module.hpp"
#include "lamppp/tensor/data_type.hpp"

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