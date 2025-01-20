#include "engine.h"

Value::Value(double data, unordered_set<shared_ptr<Value>> children,
             char op, double grad, const std::string &label)
    : data(data), grad(0.0), op(op), label(label) {}

Value Value::operator+(const Value& other) const {
    unordered_set<shared_ptr<Value>> out_prev = {
        std::make_shared<Value>(this),
        std::make_shared<Value>(other)
    };
    Value out = Value(this.data + other.data, out_prev, '+');
    out.backward = [&]() {
        this.grad += 1.0 * out.grad;
        other.grad += 1.0 * out.grad;
    };
    return out;
}

Value Value::operator-(const Value& other) const {
    Value out = this + (-1*other);
    return out;
}

Value Value::operator*(const Value& other) const {
    unordered_set<shared_ptr<Value>> out_prev = {
        std::make_shared<Value>(this),
        std::make_shared<Value>(other)
    };
    Value out = Value(this.data * other.data, out_prev, '*');
    out.backward = [&]() {
        this.grad += other.data * out.grad;
        other.grad += this.data * out.grad;
    };
    return out;
}

Value Value::operator/(const Value& other) const {
    Value out = this * other.pow(-1);
    return out;
}

Value Value::pow(const double pwr) const {
    unordered_set<shared_ptr<Value>> out_prev = {
        std::make_shared<Value>(this)
    };
    Value out = Value(pow(this.data, pwr), out_prev, '^');
    out.backward = [&]() {
        this.grad += (pwr * pow(this.data, pwr - 1)) * out.grad;
    };
    return out;
}

Value Value::exp() const {
    unordered_set<shared_ptr<Value>> out_prev = {
        std::make_shared<Value>(this)
    };
    Value out = Value(exp(this.data), out_prev, 'e');
    out.backward = [&]() {
        this.grad += out.data * out.grad;
    };
    return out;
}

Value Value::tanh() const {
    Value out;
    return out;
}

Value Value::relu() const {
    Value out;
    return out;
}

void Value::backward() {

}

void Value::internalTopoSort(const Value* node) {
}

std::ostream& operator<<(std::ostream& os, const Value& obj) {
    return os;
}

