#pragma once

#ifndef _FUNCTION_H_
#define _FUNCTION_H_

#include <cassert>
#include <memory>
#include "variable.h"

namespace autograd {

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

}  // namespace autograd

#endif  // _FUNCTION_H_