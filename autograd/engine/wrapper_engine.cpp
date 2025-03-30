#include "wrapper_engine.h"

#include <utility>

SharedValue::SharedValue() : value_(std::make_shared<Value>(0.0)) {}
SharedValue::SharedValue(double data) : value_(std::make_shared<Value>(data)) {}
SharedValue::SharedValue(std::shared_ptr<Value> value_) : value_(std::move(value_)) {}

double SharedValue::getData() const {
  return value_->data;
}
double SharedValue::getGrad() const {
  return value_->grad;
}
std::shared_ptr<Value> SharedValue::getPtr() const {
  return value_;
}

SharedValue SharedValue::operator+(const SharedValue& other) const {
  return SharedValue(value_ + other.value_);
}

SharedValue SharedValue::operator-(const SharedValue& other) const {
  return SharedValue(value_ - other.value_);
}

SharedValue SharedValue::operator*(const SharedValue& other) const {
  return SharedValue(value_ * other.value_);
}

SharedValue SharedValue::operator/(const SharedValue& other) const {
  return SharedValue(value_ / other.value_);
}

SharedValue& SharedValue::operator+=(const SharedValue& other) {
  value_ = value_ + other.value_;
  return *this;
}

SharedValue& SharedValue::operator-=(const SharedValue& other) {
  value_ = value_ - other.value_;
  return *this;
}

SharedValue& SharedValue::operator*=(const SharedValue& other) {
  value_ = value_ * other.value_;
  return *this;
}

SharedValue& SharedValue::operator/=(const SharedValue& other) {
  value_ = value_ / other.value_;
  return *this;
}

// Operations with scalar values
SharedValue SharedValue::operator+(double scalar) const {
  return SharedValue(value_ + scalar);
}

SharedValue SharedValue::operator-(double scalar) const {
  return SharedValue(value_ - scalar);
}

SharedValue SharedValue::operator*(double scalar) const {
  return SharedValue(value_ * scalar);
}

SharedValue SharedValue::operator/(double scalar) const {
  return SharedValue(value_ / scalar);
}

SharedValue& SharedValue::operator+=(double scalar) {
  value_ = value_ + scalar;
  return *this;
}

SharedValue& SharedValue::operator-=(double scalar) {
  value_ = value_ - scalar;
  return *this;
}

SharedValue& SharedValue::operator*=(double scalar) {
  value_ = value_ * scalar;
  return *this;
}

SharedValue& SharedValue::operator/=(double scalar) {
  value_ = value_ / scalar;
  return *this;
}

// Comparison operators
bool SharedValue::operator<(const SharedValue& other) const {
  return value_->data < other.value_->data;
}

bool SharedValue::operator>(const SharedValue& other) const {
  return value_->data > other.value_->data;
}

bool SharedValue::operator==(const SharedValue& other) const {
  return value_->data == other.value_->data;
}

bool SharedValue::operator!=(const SharedValue& other) const {
  return value_->data != other.value_->data;
}

bool SharedValue::operator<=(const SharedValue& other) const {
  return value_->data <= other.value_->data;
}

bool SharedValue::operator>=(const SharedValue& other) const {
  return value_->data >= other.value_->data;
}

bool SharedValue::operator<(double scalar) const {
  return value_->data < scalar;
}

bool SharedValue::operator>(double scalar) const {
  return value_->data > scalar;
}

bool SharedValue::operator==(double scalar) const {
  return value_->data == scalar;
}

bool SharedValue::operator!=(double scalar) const {
  return value_->data != scalar;
}

bool SharedValue::operator<=(double scalar) const {
  return value_->data <= scalar;
}

bool SharedValue::operator>=(double scalar) const {
  return value_->data >= scalar;
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
  return SharedValue(value_->exp());
}

SharedValue SharedValue::log() const {
  return SharedValue(value_->log());
}

SharedValue SharedValue::pow(const SharedValue& exponent) const {
  return SharedValue(value_->pow(exponent.value_));
}

SharedValue SharedValue::tanh() const {
  return SharedValue(value_->tanh());
}

SharedValue SharedValue::relu() const {
  std::cout << "ppp" << std::endl;
  return SharedValue(value_->relu());
}

void SharedValue::backprop() {
  value_->backprop();
}

SharedValue operator+(double scalar, const SharedValue& value) {
  return value + scalar;
}

SharedValue operator-(double scalar, const SharedValue& value) {
  std::shared_ptr<Value> scalar_value = std::make_shared<Value>(scalar);
  return SharedValue(scalar_value - value.getPtr());
}

SharedValue operator*(double scalar, const SharedValue& value) {
  return value * scalar;
}

SharedValue operator/(double scalar, const SharedValue& value) {
  std::shared_ptr<Value> scalar_value = std::make_shared<Value>(scalar);
  return SharedValue(scalar_value / value.getPtr());
}

std::ostream& operator<<(std::ostream& os, const SharedValue& obj) {
  os << *obj.getPtr();
  return os;
}
