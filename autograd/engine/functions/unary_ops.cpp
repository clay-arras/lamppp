#include "autograd/engine/variable.h"
#include "unary_ops.h"
#include <cassert>
#include <cmath>

variable_list Exponential::apply(const variable_list& inputs) {
    assert(inputs.size() == 1);
    const Variable& self = inputs[0];

    Variable result = Variable(std::exp(self.data()));
    return {result};
}

variable_list Logarithm::apply(const variable_list& inputs) {
    assert(inputs.size() == 1);
    const Variable& self = inputs[0];

    Variable result = Variable(std::log(self.data()));
    return {result};
}

variable_list ReLU::apply(const variable_list& inputs) {
    assert(inputs.size() == 1);
    const Variable& self = inputs[0];

    Variable result = Variable(std::max(self.data(), 0.0F));
    return {result};
}