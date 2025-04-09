#include <autograd/engine/function.h>

struct Add : public Function {
    variable_list apply(const variable_list& inputs) override;
};

struct Multiply : public Function {
    variable_list apply(const variable_list& inputs) override;
};