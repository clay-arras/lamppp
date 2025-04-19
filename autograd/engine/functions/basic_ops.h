#pragma once

#ifndef _BASIC_OPS_H_
#define _BASIC_OPS_H_

#include <autograd/engine/function.h>
#include <autograd/engine/forward_function.h>

namespace autograd {

struct AddBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct SubtractBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct MultiplyBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct DivideBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Add : public ForwardFunction<Add> {
  using DefaultBackward = AddBackward;
  static Tensor execute(const variable_list& inputs);
};

struct Subtract : public ForwardFunction<Subtract> {
  using DefaultBackward = SubtractBackward;
  static Tensor execute(const variable_list& inputs);
};

struct Multiply : public ForwardFunction<Multiply> {
  using DefaultBackward = MultiplyBackward;
  static Tensor execute(const variable_list& inputs);
};

struct Divide : public ForwardFunction<Divide> {
  using DefaultBackward = DivideBackward;
  static Tensor execute(const variable_list& inputs);
};

}  // namespace autograd

#endif  // _BASIC_OPS_H_