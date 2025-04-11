#include <autograd/engine/function.h>

struct Add : public Function {
    variable_list apply(const variable_list& inputs) override;
};

struct AddBackward : public Function {
    variable_list apply(const variable_list& gradOutputs) override;
};

struct Subtract : public Function {
    variable_list apply(const variable_list& inputs) override;
};

struct SubtractBackward : public Function {
    variable_list apply(const variable_list& gradOutputs) override;
};

struct Multiply : public Function {
    variable_list apply(const variable_list& inputs) override;
};

struct MultiplyBackward : public Function {
    variable_list apply(const variable_list& gradOutputs) override;
};

struct Divide : public Function {
    variable_list apply(const variable_list& inputs) override;
};

struct DivideBackward : public Function {
    variable_list apply(const variable_list& gradOutputs) override;
};