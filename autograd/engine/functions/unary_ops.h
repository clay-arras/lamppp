#include <autograd/engine/function.h>

namespace autograd {

struct Exponential : public Function {
  variable_list apply(const variable_list& inputs) override;
};

struct Logarithm : public Function {
  variable_list apply(const variable_list& inputs) override;
};

struct ReLU : public Function {
  variable_list apply(const variable_list& inputs) override;
};

struct ExponentialBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct LogarithmBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct ReLUBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

}