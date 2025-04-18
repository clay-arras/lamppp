#pragma once

#ifndef _FORWARD_FUNCTION_H_
#define _FORWARD_FUNCTION_H_

#include "function.h"
#include "variable.h"
#include <numeric>

namespace autograd {

template <typename Derived>
struct ForwardFunction : public Function {
  static bool requires_grad(const variable_list& variables) {
    return std::accumulate(variables.begin(), variables.end(), false, [](bool accumulated, const Variable& b) { 
      return accumulated || b.requires_grad(); 
    });
  }

  variable_list apply(const variable_list& inputs) override { // TODO(nlin): is this even necessary
    throw std::runtime_error("Forward function should not be called without template args");
    return inputs; // stub
  }

  template <typename ...Args>
  variable_list apply(const variable_list& inputs, Args&&... args) {
    Variable result = Variable(static_cast<Derived*>(this)->execute(inputs), 
        requires_grad(inputs));
    auto backward_fn = std::make_shared<typename Derived::DefaultBackward>(std::forward<Args>(args)...);
    backward_fn->saved_inputs =
        std::make_unique<variable_list>(inputs);
    result.set_grad_fn(backward_fn);

    return {result};
  }
};

}  // namespace autograd

#endif  // _FORWARD_FUNCTION_H_