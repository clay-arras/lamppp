#include <autograd/engine/function.h>

struct Exponential : public Function {
  variable_list apply(const variable_list& inputs) override;
};

struct Logarithm : public Function {
  variable_list apply(const variable_list& inputs) override;
};

struct ReLU : public Function {
  variable_list apply(const variable_list& inputs) override;
};