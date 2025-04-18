#pragma once

#ifndef _REDUCT_OPS_H_
#define _REDUCT_OPS_H_

#include "autograd/engine/function.h"
#include "autograd/engine/forward_function.h"

namespace autograd {

struct SummationBackward : public Function {
  int axis;
  explicit SummationBackward(int axis) : axis(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct MaximumBackward : public Function {
  int axis;
  explicit MaximumBackward(int axis) : axis(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Summation : public ForwardFunction<Summation> {
  using DefaultBackward = SummationBackward;
  int axis;
  explicit Summation(int axis) : axis(axis) {}
  Tensor execute(const variable_list& inputs) const;
};

struct Maximum : public ForwardFunction<Maximum> {
  using DefaultBackward = MaximumBackward;
  int axis;
  explicit Maximum(int axis) : axis(axis) {}
  Tensor execute(const variable_list& inputs) const;
};

}  // namespace autograd

#endif  // _REDUCT_OPS_H_
