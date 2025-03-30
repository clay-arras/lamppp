#include "wrapper_engine.h"

SharedValue::SharedValue() : value(std::make_shared<Value>(0.0)) {}
SharedValue::SharedValue(double data) : value(std::make_shared<Value>(data)) {}
SharedValue::SharedValue(std::shared_ptr<Value> value) : value(value) {}

double SharedValue::getData() const { return value->data; }
double SharedValue::getGrad() const { return value->grad; }
std::shared_ptr<Value> SharedValue::getPtr() const { return value; }

SharedValue SharedValue::operator+(const SharedValue& other) const {
    return SharedValue(value + other.value);
}

SharedValue SharedValue::operator-(const SharedValue& other) const {
    return SharedValue(value - other.value);
}

SharedValue SharedValue::operator*(const SharedValue& other) const {
    return SharedValue(value * other.value);
}

SharedValue SharedValue::operator/(const SharedValue& other) const {
    return SharedValue(value / other.value);
}

SharedValue& SharedValue::operator+=(const SharedValue& other) {
    value = value + other.value;
    return *this;
}

SharedValue& SharedValue::operator-=(const SharedValue& other) {
    value = value - other.value;
    return *this;
}

SharedValue& SharedValue::operator*=(const SharedValue& other) {
    value = value * other.value;
    return *this;
}

SharedValue& SharedValue::operator/=(const SharedValue& other) {
    value = value / other.value;
    return *this;
}

// Operations with scalar values
SharedValue SharedValue::operator+(double scalar) const {
    return SharedValue(value + scalar);
}

SharedValue SharedValue::operator-(double scalar) const {
    return SharedValue(value - scalar);
}

SharedValue SharedValue::operator*(double scalar) const {
    return SharedValue(value * scalar);
}

SharedValue SharedValue::operator/(double scalar) const {
    return SharedValue(value / scalar);
}

SharedValue& SharedValue::operator+=(double scalar) {
    value = value + scalar;
    return *this;
}

SharedValue& SharedValue::operator-=(double scalar) {
    value = value - scalar;
    return *this;
}

SharedValue& SharedValue::operator*=(double scalar) {
    value = value * scalar;
    return *this;
}

SharedValue& SharedValue::operator/=(double scalar) {
    value = value / scalar;
    return *this;
}

SharedValue SharedValue::exp() const {
    return SharedValue(::exp(value));
}

SharedValue SharedValue::log() const {
    return SharedValue(::log(value));
}

SharedValue SharedValue::pow(const SharedValue& exponent) const {
    return SharedValue(value->pow(exponent.value));
}

SharedValue SharedValue::tanh() const {
    return SharedValue(::tanh(value));
}

SharedValue SharedValue::relu() const {
    return SharedValue(::relu(value));
}

void SharedValue::backprop() {
    value->backprop();
}

SharedValue operator+(double scalar, const SharedValue& value) {
    return value + scalar;
}

SharedValue operator-(double scalar, const SharedValue& value) {
    std::shared_ptr<Value> scalarValue = std::make_shared<Value>(scalar);
    return SharedValue(scalarValue - value.getPtr());
}

SharedValue operator*(double scalar, const SharedValue& value) {
    return value * scalar;
}

SharedValue operator/(double scalar, const SharedValue& value) {
    std::shared_ptr<Value> scalarValue = std::make_shared<Value>(scalar);
    return SharedValue(scalarValue / value.getPtr());
}

std::ostream& operator<<(std::ostream& os, const SharedValue& obj) {
    os << *obj.getPtr();
    return os;
}
