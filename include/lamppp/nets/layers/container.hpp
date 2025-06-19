#pragma once

#include <utility>
#include "lamppp/autograd/variable.hpp"
#include "lamppp/nets/module.hpp"
#include "lamppp/nets/any.hpp"
#include "lamppp/tensor/scalar.hpp"

namespace lmp::nets {

class SequentialImpl : public ModuleImpl {
 public:
  explicit SequentialImpl(std::vector<AnyModule> layers)
      : layers_(std::move(layers)) {};
  std::any forward(const std::vector<std::any>& in) const;

 private:
  std::vector<AnyModule> layers_;
};

struct Sequential : public ModuleCRTP<SequentialImpl> {
  explicit Sequential(std::vector<AnyModule> layers);
};

}

/*

idea 1: template input and output
idea 2: just return anyValues and trust user to decode it

*/