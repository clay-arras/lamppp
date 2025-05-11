#pragma once

#include <numeric>
#include "function.hpp"
#include "variable.hpp"

namespace autograd {

template <typename Derived>
struct ForwardFunction : public Function {
  static bool requires_grad(const variable_list& variables) {
    return std::accumulate(variables.begin(), variables.end(), false,
                           [](bool accumulated, const Variable& b) {
                             return accumulated || b.requires_grad();
                           });
  }

  variable_list apply(const variable_list& inputs) override {
    throw std::runtime_error(
        "Forward function should not be called without template args");
    return {};
  }

  template <typename... Args>
  variable_list apply(const variable_list& inputs, Args&&... args) {
    Tensor tmp = static_cast<Derived*>(this)->execute(inputs);
    // std::cout << "TEN RESULT IN FF: " << tmp << std::endl;
    Variable result = Variable(tmp, requires_grad(inputs));
    // std::cout << "RESULT IN FF: " << result << std::endl;
    auto backward_fn = std::make_shared<typename Derived::DefaultBackward>(
        std::forward<Args>(args)...);
    backward_fn->saved_inputs = std::make_unique<variable_list>(inputs);
    result.set_grad_fn(backward_fn);

    return {result};
  }
};

}  // namespace autograd
