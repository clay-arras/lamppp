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

// Comparison operators
bool SharedValue::operator<(const SharedValue& other) const {
    return value->data < other.value->data;
}

bool SharedValue::operator>(const SharedValue& other) const {
    return value->data > other.value->data;
}

bool SharedValue::operator==(const SharedValue& other) const {
    return value->data == other.value->data;
}

bool SharedValue::operator!=(const SharedValue& other) const {
    return value->data != other.value->data;
}

bool SharedValue::operator<=(const SharedValue& other) const {
    return value->data <= other.value->data;
}

bool SharedValue::operator>=(const SharedValue& other) const {
    return value->data >= other.value->data;
}

// Comparison with scalar
bool SharedValue::operator<(double scalar) const {
    return value->data < scalar;
}

bool SharedValue::operator>(double scalar) const {
    return value->data > scalar;
}

bool SharedValue::operator==(double scalar) const {
    return value->data == scalar;
}

bool SharedValue::operator!=(double scalar) const {
    return value->data != scalar;
}

bool SharedValue::operator<=(double scalar) const {
    return value->data <= scalar;
}

bool SharedValue::operator>=(double scalar) const {
    return value->data >= scalar;
}

bool operator<(double scalar, const SharedValue& value) {
    return scalar < value.getData();
}

bool operator>(double scalar, const SharedValue& value) {
    return scalar > value.getData();
}

bool operator==(double scalar, const SharedValue& value) {
    return scalar == value.getData();
}

bool operator!=(double scalar, const SharedValue& value) {
    return scalar != value.getData();
}

bool operator<=(double scalar, const SharedValue& value) {
    return scalar <= value.getData();
}

bool operator>=(double scalar, const SharedValue& value) {
    return scalar >= value.getData();
}

SharedValue SharedValue::exp() const {
    return SharedValue(value->exp());
}

SharedValue SharedValue::log() const {
    return SharedValue(value->log());
}

SharedValue SharedValue::pow(const SharedValue& exponent) const {
    return SharedValue(value->pow(exponent.value));
}

SharedValue SharedValue::tanh() const {
    return SharedValue(value->tanh());
}

SharedValue SharedValue::relu() const {
    std::cout << "ppp" << std::endl;
    return SharedValue(value->relu());
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
