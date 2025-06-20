#pragma once

#include "lamppp/autograd/variable.hpp"
#include "lamppp/nets/module.hpp"

namespace lmp::nets {

class ReLUImpl : public ModuleImpl {
 public:
  ReLUImpl() = default;
  autograd::Variable forward(const autograd::Variable& x) const;
};
LMP_DEFINE_MODULE(ReLU);

}