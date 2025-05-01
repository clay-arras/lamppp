#pragma once

#ifndef _BINARY_OPS_H_
#define _BINARY_OPS_H_

#include "autograd/engine/forward_function.hpp"
#include "autograd/engine/function.hpp"

namespace autograd {

struct EqualBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct LessBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct LessEqualBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct NotEqualBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct GreaterBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct GreaterEqualBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Equal : public ForwardFunction<Equal> {
  using DefaultBackward = EqualBackward;
  static Tensor execute(const variable_list& inputs);
};

struct Less : public ForwardFunction<Less> {
  using DefaultBackward = LessBackward;
  static Tensor execute(const variable_list& inputs);
};

struct LessEqual : public ForwardFunction<LessEqual> {
  using DefaultBackward = LessEqualBackward;
  static Tensor execute(const variable_list& inputs);
};

struct NotEqual : public ForwardFunction<NotEqual> {
  using DefaultBackward = NotEqualBackward;
  static Tensor execute(const variable_list& inputs);
};

struct Greater : public ForwardFunction<Greater> {
  using DefaultBackward = GreaterBackward;
  static Tensor execute(const variable_list& inputs);
};

struct GreaterEqual : public ForwardFunction<GreaterEqual> {
  using DefaultBackward = GreaterEqualBackward;
  static Tensor execute(const variable_list& inputs);
};

}  // namespace autograd

#endif  // _BINARY_OPS_H_
