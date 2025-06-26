#pragma once

#include <utility>
#include "lamppp/autograd/variable.hpp"
#include "lamppp/nets/module.hpp"
#include "lamppp/nets/any.hpp"
#include "lamppp/tensor/data_type.hpp"

namespace lmp::nets {

class SequentialImpl : public ModuleImpl {
 public:
  explicit SequentialImpl(std::vector<AnyModule> layers);
  std::any forward(const std::vector<std::any>& in) const;

 private:
  std::vector<AnyModule> layers_;
};
LMP_DEFINE_MODULE(Sequential);

}