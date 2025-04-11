#include "basic_ops.h"
#include "autograd/engine/variable.h"
#include <cassert>
#include <memory>
#include "autograd/engine/function.h"

variable_list AddBackward::apply(const variable_list& gradOutputs) { 
    assert(gradOutputs.size() == 1); 
    const Variable& grad = gradOutputs[0];
    variable_list grad_inputs = {grad, grad};
    return grad_inputs;
}


variable_list SubtractBackward::apply(const variable_list& gradOutputs) { 
    assert(gradOutputs.size() == 1); 
    const Variable& grad = gradOutputs[0];
    variable_list grad_inputs = {grad, grad * -1.0F};
    return grad_inputs;
}

variable_list MultiplyBackward::apply(const variable_list& gradOutputs) { 
    assert(gradOutputs.size() == 1); 
    const Variable& grad = gradOutputs[0];
    const Variable& self = (*saved_inputs)[0];
    const Variable& other = (*saved_inputs)[1];

    variable_list grad_inputs = {other * grad, self * grad};
    return grad_inputs;
}

variable_list DivideBackward::apply(const variable_list& gradOutputs) { 
    assert(gradOutputs.size() == 1); 
    const Variable& grad = gradOutputs[0];
    const Variable& self = (*saved_inputs)[0];
    const Variable& other = (*saved_inputs)[1];

    variable_list grad_inputs = {grad / other, (grad * self * -1.0F) / (other * other)};
    return grad_inputs;
}

variable_list Add::apply(const variable_list& inputs) {
    assert(inputs.size() == 2);
    const Variable& self = inputs[0];
    const Variable& other = inputs[1]; 

    Variable result = Variable(self.data() + other.data());
    auto backward_fn = std::make_shared<AddBackward>();
    backward_fn->saved_inputs = std::make_unique<variable_list>(variable_list{self, other});
    result.set_grad_fn(backward_fn);
    return {result};
}

variable_list Subtract::apply(const variable_list& inputs) {
    assert(inputs.size() == 2);
    const Variable& self = inputs[0];
    const Variable& other = inputs[1];

    Variable result = Variable(self.data() - other.data());
    auto backward_fn = std::make_shared<SubtractBackward>();
    backward_fn->saved_inputs = std::make_unique<variable_list>(variable_list{self, other});
    result.set_grad_fn(backward_fn);
    return {result};
}

variable_list Multiply::apply(const variable_list& inputs) {
    assert(inputs.size() == 2);
    const Variable& self = inputs[0];
    const Variable& other = inputs[1];

    Variable result = Variable(self.data() * other.data());
    auto backward_fn = std::make_shared<MultiplyBackward>();
    backward_fn->saved_inputs = std::make_unique<variable_list>(variable_list{self, other});
    result.set_grad_fn(backward_fn);
    return {result};
}

variable_list Divide::apply(const variable_list& inputs) {
    assert(inputs.size() == 2);
    const Variable& self = inputs[0];
    const Variable& other = inputs[1];

    Variable result = Variable(self.data() / other.data());
    auto backward_fn = std::make_shared<DivideBackward>();
    backward_fn->saved_inputs = std::make_unique<variable_list>(variable_list{self, other});
    result.set_grad_fn(backward_fn);
    return {result};
}
