#include "autograd/engine/function.h"

namespace autograd {

struct MatrixMultiplication : public Function {
  variable_list apply(const variable_list& inputs) override;
};

struct Transpose : public Function {
  variable_list apply(const variable_list& inputs) override;
};

struct MatrixMultiplicationBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

struct TransposeBackward : public Function {
  variable_list apply(const variable_list& gradOutputs) override;
};

}