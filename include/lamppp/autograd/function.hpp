#pragma once

#include <cassert>
#include <memory>
#include "variable.hpp"

namespace lmp::autograd {

class Variable;
struct Function;

struct Function : public std::enable_shared_from_this<Function> {
  std::unique_ptr<variable_list> saved_inputs;

  Function() = default;
  virtual ~Function() = default;

  virtual variable_list apply(const variable_list& inputs) = 0;
  variable_list operator()(const variable_list& inputs) {
    return apply(inputs);
  }
};

}  // namespace lmp::autograd
