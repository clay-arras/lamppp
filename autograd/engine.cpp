#include "engine.h"

Value::Value(double data, std::unordered_set<std::shared_ptr<Value>> children,
             char op, double grad, const std::string &label)
    : data(data), grad(grad), op(op), label(label), prev(children), backward([]() -> void {}) {}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a->operator+(b); 
} 

std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
    std::unordered_set<std::shared_ptr<Value>> prev_out;
    prev_out.insert(shared_from_this());
    prev_out.insert(other);
    
    std::shared_ptr<Value> out = std::make_shared<Value>(this->data + other->data, prev_out, '+');

    out->backward = [self=shared_from_this(), other, out]() {
        self->grad += 1.0 * out->grad;
        other->grad += 1.0 * out->grad;
    };
    return out;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a->operator-(b);
}

std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {
    return shared_from_this() + (other * std::make_shared<Value>(-1));
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a->operator*(b);
} 

std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {
    std::unordered_set<std::shared_ptr<Value>> prev_out;
    prev_out.insert(shared_from_this());
    prev_out.insert(other);
    auto out = std::make_shared<Value>(this->data * other->data, prev_out, '*');

    out->backward = [self=shared_from_this(), other, out]() {
        self->grad += other->data * out->grad;
        other->grad += self->data * out->grad;
    };
    return out;
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a->operator/(b);
}

std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value>& other) {
    return shared_from_this() * other->pow(-1);
}

std::shared_ptr<Value> Value::pow(const double pwr) {
    std::unordered_set<std::shared_ptr<Value>> prev_out;
    prev_out.insert(shared_from_this());
    auto out = std::make_shared<Value>(std::pow(this->data, pwr), prev_out, '^');

    out->backward = [self=shared_from_this(), pwr, out]() {
        self->grad += (pwr * std::pow(self->data, pwr - 1)) * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::exp() {
    std::unordered_set<std::shared_ptr<Value>> prev_out;
    prev_out.insert(shared_from_this());
    auto out = std::make_shared<Value>(std::exp(this->data), prev_out, 'e');

    out->backward = [self=shared_from_this(), out]() {
        self->grad += out->data * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::tanh() {
    auto exp = (std::make_shared<Value>(2) * shared_from_this())->exp();
    auto one = std::make_shared<Value>(1);
    return (exp - one)/(exp + one);
}

std::shared_ptr<Value> Value::relu() {
    // Value out;
    // return out;
    return nullptr;
}

void Value::backprop() {
    std::vector<std::shared_ptr<Value>> topo = internalTopoSort();
    this->grad = 1.0;
    for (std::shared_ptr<Value> node : topo) {
        node->backward();
    }
}

std::vector<std::shared_ptr<Value>> Value::internalTopoSort() {
    std::unordered_set<std::shared_ptr<Value>> visited;
    std::vector<std::shared_ptr<Value>> topo;

    std::function<void(std::shared_ptr<Value>)> dfs = [&](std::shared_ptr<Value> v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (const auto& child : v->prev) {
                dfs(child);
            }
            topo.push_back(v);
        }
    };

    dfs(shared_from_this());
    std::reverse(topo.begin(), topo.end());
    return topo;
}

std::ostream& operator<<(std::ostream& os, const Value& obj) {
    os << "Value(data=" << obj.data << ", grad=" << obj.grad << ")";
    return os;
}

