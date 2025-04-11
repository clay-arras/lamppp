#include "unary_ops.h"
#include <cassert>
#include <cmath>
#include "autograd/engine/variable.h"

variable_list Exponential::apply(
    const variable_list&
        inputs) {  // TODO(nlin): need to fix these with backward
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  Variable result = Variable(self.data().exp());
  return {result};
}

variable_list Logarithm::apply(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  Variable result = Variable(self.data().log());
  return {result};
}

variable_list ReLU::apply(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  Variable result = Variable(self.data().relu());
  return {result};
}