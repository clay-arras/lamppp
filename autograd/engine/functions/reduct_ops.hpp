#pragma once

#ifndef _REDUCT_OPS_H_
#define _REDUCT_OPS_H_

#include "autograd/engine/forward_function.hpp"
#include "autograd/engine/function.hpp"

namespace autograd {

struct SummationBackward : public Function {
  size_t axis;
  explicit SummationBackward(size_t axis) : axis(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct MaximumBackward : public Function {
  size_t axis;
  explicit MaximumBackward(size_t axis) : axis(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};

struct Summation : public ForwardFunction<Summation> {
  using DefaultBackward = SummationBackward;
  size_t axis;
  explicit Summation(size_t axis) : axis(axis) {}
  Tensor execute(const variable_list& inputs) const;
};

struct Maximum : public ForwardFunction<Maximum> {
  using DefaultBackward = MaximumBackward;
  size_t axis;
  explicit Maximum(size_t axis) : axis(axis) {}
  Tensor execute(const variable_list& inputs) const;
};

}  // namespace autograd

#endif  // _REDUCT_OPS_H_
