#include "variable.h"

Variable Variable::operator+(const Variable& other) const {
  return Variable(this->data() + other.data());
}

Variable Variable::operator-(const Variable& other) const {
  return Variable(this->data() - other.data());
}

Variable Variable::operator*(const Variable& other) const {
  return Variable(this->data() * other.data());
}

Variable Variable::operator/(const Variable& other) const {
  return Variable(this->data() / other.data());
}

Variable& Variable::operator+=(const Variable& other) {
  this->data() += other.data();
  return *this;
}

Variable& Variable::operator-=(const Variable& other) {
  this->data() -= other.data();
  return *this;
}

Variable& Variable::operator*=(const Variable& other) {
  this->data() *= other.data();
  return *this;
}

Variable& Variable::operator/=(const Variable& other) {
  this->data() /= other.data();
  return *this;
}

bool Variable::operator<(const Variable& other) const {
  return this->data() < other.data();
}

bool Variable::operator>(const Variable& other) const {
  return this->data() > other.data();
}

bool Variable::operator==(const Variable& other) const {
  return this->data() == other.data();
}

bool Variable::operator!=(const Variable& other) const {
  return this->data() != other.data();
}

bool Variable::operator<=(const Variable& other) const {
  return this->data() <= other.data();
}

bool Variable::operator>=(const Variable& other) const {
  return this->data() >= other.data();
}