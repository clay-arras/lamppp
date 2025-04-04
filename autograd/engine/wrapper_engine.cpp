#include "wrapper_engine.h"

#include <utility>

/**
 * @brief Default constructor for SharedValue.
 * Initializes the SharedValue with a default value of 0.0.
 */
SharedValue::SharedValue() : value_(std::make_shared<Value>(0.0)) {}

/**
 * @brief Constructs a SharedValue from a double.
 * @param data The double value to initialize the SharedValue.
 */
SharedValue::SharedValue(double data) : value_(std::make_shared<Value>(data)) {}

/**
 * @brief Constructs a SharedValue from a shared pointer to a Value.
 * @param value_ The shared pointer to a Value object.
 */
SharedValue::SharedValue(std::shared_ptr<Value> value_) : value_(std::move(value_)) {}

/**
 * @brief Retrieves the data stored in the SharedValue.
 * @return The double data contained in the SharedValue.
 */
double SharedValue::getData() const {
  return value_->data;
}

/**
 * @brief Retrieves the gradient associated with the SharedValue.
 * @return The gradient as a double.
 */
double SharedValue::getGrad() const {
  return value_->grad;
}

/**
 * @brief Retrieves the shared pointer to the underlying Value.
 * @return A shared pointer to the Value object.
 */
std::shared_ptr<Value> SharedValue::getPtr() const {
  return value_;
}

/**
 * @brief Addition operator for SharedValue.
 * @param other The other SharedValue to add.
 * @return A new SharedValue that is the sum of this and the other.
 */
SharedValue SharedValue::operator+(const SharedValue& other) const {
  return SharedValue(value_ + other.value_);
}

/**
 * @brief Subtraction operator for SharedValue.
 * @param other The other SharedValue to subtract.
 * @return A new SharedValue that is the difference of this and the other.
 */
SharedValue SharedValue::operator-(const SharedValue& other) const {
  return SharedValue(value_ - other.value_);
}

/**
 * @brief Multiplication operator for SharedValue.
 * @param other The other SharedValue to multiply.
 * @return A new SharedValue that is the product of this and the other.
 */
SharedValue SharedValue::operator*(const SharedValue& other) const {
  return SharedValue(value_ * other.value_);
}

/**
 * @brief Division operator for SharedValue.
 * @param other The other SharedValue to divide by.
 * @return A new SharedValue that is the quotient of this and the other.
 */
SharedValue SharedValue::operator/(const SharedValue& other) const {
  return SharedValue(value_ / other.value_);
}

/**
 * @brief Addition assignment operator for SharedValue.
 * @param other The other SharedValue to add.
 * @return A reference to this SharedValue after addition.
 */
SharedValue& SharedValue::operator+=(const SharedValue& other) {
  value_ = value_ + other.value_;
  return *this;
}

/**
 * @brief Subtraction assignment operator for SharedValue.
 * @param other The other SharedValue to subtract.
 * @return A reference to this SharedValue after subtraction.
 */
SharedValue& SharedValue::operator-=(const SharedValue& other) {
  value_ = value_ - other.value_;
  return *this;
}

/**
 * @brief Multiplication assignment operator for SharedValue.
 * @param other The other SharedValue to multiply.
 * @return A reference to this SharedValue after multiplication.
 */
SharedValue& SharedValue::operator*=(const SharedValue& other) {
  value_ = value_ * other.value_;
  return *this;
}

/**
 * @brief Division assignment operator for SharedValue.
 * @param other The other SharedValue to divide by.
 * @return A reference to this SharedValue after division.
 */
SharedValue& SharedValue::operator/=(const SharedValue& other) {
  value_ = value_ / other.value_;
  return *this;
}

/**
 * @brief Addition operator for SharedValue and a scalar.
 * @param scalar The scalar value to add.
 * @return A new SharedValue that is the sum of this and the scalar.
 */
SharedValue SharedValue::operator+(double scalar) const {
  return SharedValue(value_ + scalar);
}

/**
 * @brief Subtraction operator for SharedValue and a scalar.
 * @param scalar The scalar value to subtract.
 * @return A new SharedValue that is the difference of this and the scalar.
 */
SharedValue SharedValue::operator-(double scalar) const {
  return SharedValue(value_ - scalar);
}

/**
 * @brief Multiplication operator for SharedValue and a scalar.
 * @param scalar The scalar value to multiply.
 * @return A new SharedValue that is the product of this and the scalar.
 */
SharedValue SharedValue::operator*(double scalar) const {
  return SharedValue(value_ * scalar);
}

/**
 * @brief Division operator for SharedValue by a scalar.
 * @param scalar The scalar value to divide by.
 * @return A new SharedValue that is the quotient of this and the scalar.
 */
SharedValue SharedValue::operator/(double scalar) const {
  return SharedValue(value_ / scalar);
}

/**
 * @brief Addition assignment operator for SharedValue and a scalar.
 * @param scalar The scalar value to add.
 * @return A reference to this SharedValue after addition.
 */
SharedValue& SharedValue::operator+=(double scalar) {
  value_ = value_ + scalar;
  return *this;
}

/**
 * @brief Subtraction assignment operator for SharedValue and a scalar.
 * @param scalar The scalar value to subtract.
 * @return A reference to this SharedValue after subtraction.
 */
SharedValue& SharedValue::operator-=(double scalar) {
  value_ = value_ - scalar;
  return *this;
}

/**
 * @brief Multiplication assignment operator for SharedValue and a scalar.
 * @param scalar The scalar value to multiply.
 * @return A reference to this SharedValue after multiplication.
 */
SharedValue& SharedValue::operator*=(double scalar) {
  value_ = value_ * scalar;
  return *this;
}

/**
 * @brief Division assignment operator for SharedValue by a scalar.
 * @param scalar The scalar value to divide by.
 * @return A reference to this SharedValue after division.
 */
SharedValue& SharedValue::operator/=(double scalar) {
  value_ = value_ / scalar;
  return *this;
}


/**
 * @brief Less than comparison operator for SharedValue.
 * @param other The other SharedValue to compare.
 * @return True if this is less than the other, false otherwise.
 */
bool SharedValue::operator<(const SharedValue& other) const {
  return value_->data < other.value_->data;
}

/**
 * @brief Greater than comparison operator for SharedValue.
 * @param other The other SharedValue to compare.
 * @return True if this is greater than the other, false otherwise.
 */
bool SharedValue::operator>(const SharedValue& other) const {
  return value_->data > other.value_->data;
}

/**
 * @brief Equality comparison operator for SharedValue.
 * @param other The other SharedValue to compare.
 * @return True if this is equal to the other, false otherwise.
 */
bool SharedValue::operator==(const SharedValue& other) const {
  return value_->data == other.value_->data;
}

/**
 * @brief Inequality comparison operator for SharedValue.
 * @param other The other SharedValue to compare.
 * @return True if this is not equal to the other, false otherwise.
 */
bool SharedValue::operator!=(const SharedValue& other) const {
  return value_->data != other.value_->data;
}

/**
 * @brief Less than or equal to comparison operator for SharedValue.
 * @param other The other SharedValue to compare.
 * @return True if this is less than or equal to the other, false otherwise.
 */
bool SharedValue::operator<=(const SharedValue& other) const {
  return value_->data <= other.value_->data;
}

/**
 * @brief Greater than or equal to comparison operator for SharedValue.
 * @param other The other SharedValue to compare.
 * @return True if this is greater than or equal to the other, false otherwise.
 */
bool SharedValue::operator>=(const SharedValue& other) const {
  return value_->data >= other.value_->data;
}

/**
 * @brief Less than comparison operator for SharedValue.
 * @param scalar The scalar value to compare.
 * @return True if this SharedValue is less than the scalar, false otherwise.
 */
bool SharedValue::operator<(double scalar) const {
  return value_->data < scalar;
}

/**
 * @brief Greater than comparison operator for SharedValue.
 * @param scalar The scalar value to compare.
 * @return True if this SharedValue is greater than the scalar, false otherwise.
 */
bool SharedValue::operator>(double scalar) const {
  return value_->data > scalar;
}

/**
 * @brief Equality comparison operator for SharedValue.
 * @param scalar The scalar value to compare.
 * @return True if this SharedValue is equal to the scalar, false otherwise.
 */
bool SharedValue::operator==(double scalar) const {
  return value_->data == scalar;
}

/**
 * @brief Inequality comparison operator for SharedValue.
 * @param scalar The scalar value to compare.
 * @return True if this SharedValue is not equal to the scalar, false otherwise.
 */
bool SharedValue::operator!=(double scalar) const {
  return value_->data != scalar;
}

/**
 * @brief Less than or equal to comparison operator for SharedValue.
 * @param scalar The scalar value to compare.
 * @return True if this SharedValue is less than or equal to the scalar, false otherwise.
 */
bool SharedValue::operator<=(double scalar) const {
  return value_->data <= scalar;
}

/**
 * @brief Greater than or equal to comparison operator for SharedValue.
 * @param scalar The scalar value to compare.
 * @return True if this SharedValue is greater than or equal to the scalar, false otherwise.
 */
bool SharedValue::operator>=(double scalar) const {
  return value_->data >= scalar;
}

/**
 * @brief Less than comparison operator for scalar and SharedValue.
 * @param scalar The scalar value to compare.
 * @param value The SharedValue to compare.
 * @return True if the scalar is less than the SharedValue, false otherwise.
 */
bool operator<(double scalar, const SharedValue& value) {
  return scalar < value.getData();
}

/**
 * @brief Greater than comparison operator for scalar and SharedValue.
 * @param scalar The scalar value to compare.
 * @param value The SharedValue to compare.
 * @return True if the scalar is greater than the SharedValue, false otherwise.
 */
bool operator>(double scalar, const SharedValue& value) {
  return scalar > value.getData();
}

/**
 * @brief Equality comparison operator for scalar and SharedValue.
 * @param scalar The scalar value to compare.
 * @param value The SharedValue to compare.
 * @return True if the scalar is equal to the SharedValue, false otherwise.
 */
bool operator==(double scalar, const SharedValue& value) {
  return scalar == value.getData();
}

/**
 * @brief Inequality comparison operator for scalar and SharedValue.
 * @param scalar The scalar value to compare.
 * @param value The SharedValue to compare.
 * @return True if the scalar is not equal to the SharedValue, false otherwise.
 */
bool operator!=(double scalar, const SharedValue& value) {
  return scalar != value.getData();
}

/**
 * @brief Less than or equal to comparison operator for scalar and SharedValue.
 * @param scalar The scalar value to compare.
 * @param value The SharedValue to compare.
 * @return True if the scalar is less than or equal to the SharedValue, false otherwise.
 */
bool operator<=(double scalar, const SharedValue& value) {
  return scalar <= value.getData();
}

/**
 * @brief Greater than or equal to comparison operator for scalar and SharedValue.
 * @param scalar The scalar value to compare.
 * @param value The SharedValue to compare.
 * @return True if the scalar is greater than or equal to the SharedValue, false otherwise.
 */
bool operator>=(double scalar, const SharedValue& value) {
  return scalar >= value.getData();
}

/**
 * @brief Exponential function for SharedValue.
 * @return A new SharedValue representing the exponential of this SharedValue.
 */
SharedValue SharedValue::exp() const {
  return SharedValue(value_->exp());
}

/**
 * @brief Logarithm function for SharedValue.
 * @return A new SharedValue representing the logarithm of this SharedValue.
 */
SharedValue SharedValue::log() const {
  return SharedValue(value_->log());
}

/**
 * @brief Power function for SharedValue.
 * @param exponent The exponent to raise this SharedValue to.
 * @return A new SharedValue representing this SharedValue raised to the given exponent.
 */
SharedValue SharedValue::pow(const SharedValue& exponent) const {
  return SharedValue(value_->pow(exponent.value_));
}

/**
 * @brief Hyperbolic tangent function for SharedValue.
 * @return A new SharedValue representing the hyperbolic tangent of this SharedValue.
 */
SharedValue SharedValue::tanh() const {
  return SharedValue(value_->tanh());
}

/**
 * @brief Rectified Linear Unit (ReLU) function for SharedValue.
 * @return A new SharedValue representing the ReLU of this SharedValue.
 */
SharedValue SharedValue::relu() const {
  std::cout << "ppp" << std::endl;
  return SharedValue(value_->relu());
}

/**
 * @brief Backpropagation function for SharedValue.
 * This function triggers the backpropagation process for the underlying value.
 */
void SharedValue::backprop() {
  value_->backprop();
}

/**
 * @brief Addition operator for scalar and SharedValue.
 * @param scalar The scalar value to add.
 * @param value The SharedValue to add to the scalar.
 * @return A new SharedValue representing the result of the addition.
 */
SharedValue operator+(double scalar, const SharedValue& value) {
  return value + scalar;
}

/**
 * @brief Subtraction operator for scalar and SharedValue.
 * @param scalar The scalar value to subtract.
 * @param value The SharedValue to subtract from the scalar.
 * @return A new SharedValue representing the result of the subtraction.
 */
SharedValue operator-(double scalar, const SharedValue& value) {
  std::shared_ptr<Value> scalar_value = std::make_shared<Value>(scalar);
  return SharedValue(scalar_value - value.getPtr());
}

/**
 * @brief Multiplication operator for scalar and SharedValue.
 * @param scalar The scalar value to multiply.
 * @param value The SharedValue to multiply by the scalar.
 * @return A new SharedValue representing the result of the multiplication.
 */
SharedValue operator*(double scalar, const SharedValue& value) {
  return value * scalar;
}

/**
 * @brief Division operator for scalar and SharedValue.
 * @param scalar The scalar value to divide.
 * @param value The SharedValue to divide by the scalar.
 * @return A new SharedValue representing the result of the division.
 */
SharedValue operator/(double scalar, const SharedValue& value) {
  std::shared_ptr<Value> scalar_value = std::make_shared<Value>(scalar);
  return SharedValue(scalar_value / value.getPtr());
}

/**
 * @brief Output stream operator for SharedValue.
 * @param os The output stream to write to.
 * @param obj The SharedValue to output.
 * @return The output stream after writing the SharedValue.
 */
std::ostream& operator<<(std::ostream& os, const SharedValue& obj) {
  os << *obj.getPtr();
  return os;
}