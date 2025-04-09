#pragma once
#ifndef _VARIABLE_H_
#define _VARIABLE_H_

#include "function.h"
#include <memory>
#include <utility>

class Function;

struct VariableImpl {
    float data;
    float grad;
    std::shared_ptr<Function> _grad_fn;    
    bool requires_grad;

    explicit VariableImpl(float data, bool requires_grad = false) 
        : data(data), requires_grad(requires_grad) {}
};

class Variable {
public:
    Variable() 
        : impl_(std::make_shared<VariableImpl>(0.0F, false)) {}
    explicit Variable(std::shared_ptr<VariableImpl> &impl) 
        : impl_(std::move(impl)) {}
    explicit Variable(float data) 
        : impl_(std::make_shared<VariableImpl>(data, false)) {}

    std::shared_ptr<VariableImpl> impl_;
    float& grad() { return impl_->grad; }
    float& data() { return impl_->data; }
    std::shared_ptr<Function>& grad_fn() { return impl_->_grad_fn; }
    bool& requires_grad() { return impl_->requires_grad; }

    const float& grad() const { return impl_->grad; }
    const float& data() const { return impl_->data; }
    const std::shared_ptr<Function>& grad_fn() const { return impl_->_grad_fn; }
    bool requires_grad() const { return impl_->requires_grad; }

    Variable operator+(const Variable& other) const;
    Variable operator-(const Variable& other) const;
    Variable operator*(const Variable& other) const;
    Variable operator/(const Variable& other) const;

    Variable& operator+=(const Variable& other);
    Variable& operator-=(const Variable& other);
    Variable& operator*=(const Variable& other);
    Variable& operator/=(const Variable& other);

    bool operator<(const Variable& other) const;
    bool operator>(const Variable& other) const;
    bool operator==(const Variable& other) const;
    bool operator!=(const Variable& other) const;
    bool operator<=(const Variable& other) const;
    bool operator>=(const Variable& other) const;
};


#endif // _VARIABLE_H_