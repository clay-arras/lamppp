#pragma once

#ifndef _UNARY_OPS_H_
#define _UNARY_OPS_H_

#include <autograd/engine/function.h>
#include <autograd/engine/forward_function.h>

namespace autograd {

struct ExponentialBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct LogarithmBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct ReLUBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Exponential : public ForwardFunction<Exponential> {
  using DefaultBackward = ExponentialBackward;
  static Tensor execute(const variable_list& inputs);
};

struct Logarithm : public ForwardFunction<Logarithm> {
  using DefaultBackward = LogarithmBackward;
  static Tensor execute(const variable_list& inputs);
};

struct ReLU : public ForwardFunction<ReLU> {
  using DefaultBackward = ReLUBackward;
  static Tensor execute(const variable_list& inputs);
};


}  // namespace autograd

#endif  // _UNARY_OPS_H_