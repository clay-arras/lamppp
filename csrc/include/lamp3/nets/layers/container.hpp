#pragma once

#include <utility>
#include "lamp3/autograd/variable.hpp"
#include "lamp3/nets/module.hpp"
#include "lamp3/nets/any.hpp"
#include "lamp3/tensor/data_type.hpp"

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