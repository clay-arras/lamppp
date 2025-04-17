#pragma once

#ifndef _BASIC_OPS_H_
#define _BASIC_OPS_H_

#include <autograd/engine/function.h>

namespace autograd {

struct Add : public Function {
  variable_list apply(const variable_list& inputs) override;
};

struct Subtract : public Function {
  variable_list apply(const variable_list& inputs) override;
};

struct Multiply : public Function {
  variable_list apply(const variable_list& inputs) override;
};

struct Divide : public Function {
  variable_list apply(const variable_list& inputs) override;
};

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

}  // namespace autograd

#endif  // _BASIC_OPS_H_