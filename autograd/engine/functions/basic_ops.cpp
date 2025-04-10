#include "basic_ops.h"
#include "autograd/engine/variable.h"
#include <cassert>
#include "autograd/engine/function.h"

variable_list Add::apply(const variable_list& inputs) { // TODO(nlin): no backward for now
    assert(inputs.size() == 2);
    const Variable& self = inputs[0];
    const Variable& other = inputs[1];

    Variable result = Variable(self.data() + other.data());
    return {result};
}

variable_list Subtract::apply(const variable_list& inputs) { // TODO(nlin): no backward for now
    assert(inputs.size() == 2);
    const Variable& self = inputs[0];
    const Variable& other = inputs[1];

    Variable result = Variable(self.data() - other.data());
    return {result};
}

variable_list Multiply::apply(const variable_list& inputs) {
    assert(inputs.size() == 2);
    const Variable& self = inputs[0];
    const Variable& other = inputs[1];

    Variable result = Variable(self.data() * other.data());
    return {result};
}

variable_list Divide::apply(const variable_list& inputs) {
    assert(inputs.size() == 2);
    const Variable& self = inputs[0];
    const Variable& other = inputs[1];

    Variable result = Variable(self.data() / other.data());
    return {result};
}