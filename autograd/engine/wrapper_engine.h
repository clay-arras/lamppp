#ifndef _WRAPPER_ENGINE_H_
#define _WRAPPER_ENGINE_H_

#include <memory>
#include "engine.h"

/**
 * @brief A class that encapsulates a shared pointer to a Value object.
 */
class SharedValue {
 private:
  std::shared_ptr<Value> value_;  ///< Pointer to the underlying Value object.

 public:
  /**
   * @brief Default constructor for SharedValue.
   */
  SharedValue();

  /**
   * @brief Constructs a SharedValue from a double.
   * @param data The double value to initialize the SharedValue.
   */
  explicit SharedValue(double data);

  /**
   * @brief Constructs a SharedValue from a shared pointer to a Value.
   * @param value The shared pointer to a Value object.
   */
  explicit SharedValue(std::shared_ptr<Value> value);

  /**
   * @brief Retrieves the data stored in the SharedValue.
   * @return The double data contained in the SharedValue.
   */
  double getData() const;

  /**
   * @brief Retrieves the gradient associated with the SharedValue.
   * @return The gradient as a double.
   */
  double getGrad() const;

  /**
   * @brief Retrieves the shared pointer to the underlying Value.
   * @return A shared pointer to the Value object.
   */
  std::shared_ptr<Value> getPtr() const;

  SharedValue operator+(
      const SharedValue& other) const;  ///< Addition operator.
  SharedValue operator-(
      const SharedValue& other) const;  ///< Subtraction operator.
  SharedValue operator*(
      const SharedValue& other) const;  ///< Multiplication operator.
  SharedValue operator/(
      const SharedValue& other) const;  ///< Division operator.

  SharedValue& operator+=(
      const SharedValue& other);  ///< Addition assignment operator.
  SharedValue& operator-=(
      const SharedValue& other);  ///< Subtraction assignment operator.
  SharedValue& operator*=(
      const SharedValue& other);  ///< Multiplication assignment operator.
  SharedValue& operator/=(
      const SharedValue& other);  ///< Division assignment operator.

  SharedValue operator+(double scalar) const;  ///< Addition with a scalar.
  SharedValue operator-(double scalar) const;  ///< Subtraction with a scalar.
  SharedValue operator*(
      double scalar) const;  ///< Multiplication with a scalar.
  SharedValue operator/(double scalar) const;  ///< Division by a scalar.

  SharedValue& operator+=(
      double scalar);  ///< Addition assignment with a scalar.
  SharedValue& operator-=(
      double scalar);  ///< Subtraction assignment with a scalar.
  SharedValue& operator*=(
      double scalar);  ///< Multiplication assignment with a scalar.
  SharedValue& operator/=(double scalar);  ///< Division assignment by a scalar.

  bool operator<(const SharedValue& other) const;  ///< Less than comparison.
  bool operator>(const SharedValue& other) const;  ///< Greater than comparison.
  bool operator==(const SharedValue& other) const;  ///< Equality comparison.
  bool operator!=(const SharedValue& other) const;  ///< Inequality comparison.
  bool operator<=(
      const SharedValue& other) const;  ///< Less than or equal to comparison.
  bool operator>=(const SharedValue& other)
      const;  ///< Greater than or equal to comparison.

  bool operator<(double scalar) const;  ///< Less than comparison with a scalar.
  bool operator>(
      double scalar) const;  ///< Greater than comparison with a scalar.
  bool operator==(double scalar) const;  ///< Equality comparison with a scalar.
  bool operator!=(
      double scalar) const;  ///< Inequality comparison with a scalar.
  bool operator<=(double scalar)
      const;  ///< Less than or equal to comparison with a scalar.
  bool operator>=(double scalar)
      const;  ///< Greater than or equal to comparison with a scalar.

  SharedValue exp() const;  ///< Exponential function.
  SharedValue log() const;  ///< Natural logarithm function.
  SharedValue pow(const SharedValue& exponent) const;  ///< Power function.
  SharedValue tanh() const;  ///< Hyperbolic tangent function.
  SharedValue relu() const;  ///< Rectified linear unit function.

  /**
   * @brief Performs backpropagation for the SharedValue.
   */
  void backprop();

  /**
   * @brief Overloads the output stream operator for SharedValue.
   * @param os The output stream.
   * @param obj The SharedValue object.
   * @return The output stream after inserting the SharedValue.
   */
  friend std::ostream& operator<<(std::ostream& os, const SharedValue& obj);
};

SharedValue operator+(double scalar,
                      const SharedValue& value);  ///< Addition with a scalar.
SharedValue operator-(
    double scalar, const SharedValue& value);  ///< Subtraction with a scalar.
SharedValue operator*(
    double scalar,
    const SharedValue& value);  ///< Multiplication with a scalar.
SharedValue operator/(double scalar,
                      const SharedValue& value);  ///< Division by a scalar.

bool operator<(
    double scalar,
    const SharedValue& value);  ///< Less than comparison with a scalar.
bool operator>(
    double scalar,
    const SharedValue& value);  ///< Greater than comparison with a scalar.
bool operator==(
    double scalar,
    const SharedValue& value);  ///< Equality comparison with a scalar.
bool operator!=(
    double scalar,
    const SharedValue& value);  ///< Inequality comparison with a scalar.
bool operator<=(
    double scalar,
    const SharedValue&
        value);  ///< Less than or equal to comparison with a scalar.
bool operator>=(
    double scalar,
    const SharedValue&
        value);  ///< Greater than or equal to comparison with a scalar.

#endif  // _WRAPPER_ENGINE_H_
