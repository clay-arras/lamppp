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

Variable Variable::operator+(double scalar) const {
    return Variable(this->data() + scalar);
}

Variable Variable::operator-(double scalar) const {
    return Variable(this->data() - scalar);
}

Variable Variable::operator*(double scalar) const {
    return Variable(this->data() * scalar);
}

Variable Variable::operator/(double scalar) const {
    return Variable(this->data() / scalar);
}

Variable& Variable::operator+=(double scalar) {
    this->data() += scalar;
    return *this;
}

Variable& Variable::operator-=(double scalar) {
    this->data() -= scalar;
    return *this;
}

Variable& Variable::operator*=(double scalar) {
    this->data() *= scalar;
    return *this;
}

Variable& Variable::operator/=(double scalar) {
    this->data() /= scalar;
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

bool Variable::operator<(double scalar) const {
    return this->data() < scalar;
}

bool Variable::operator>(double scalar) const {
    return this->data() > scalar;
}

bool Variable::operator==(double scalar) const {
    return this->data() == scalar;
}

bool Variable::operator!=(double scalar) const {
    return this->data() != scalar;
}

bool Variable::operator<=(double scalar) const {
    return this->data() <= scalar;
}

bool Variable::operator>=(double scalar) const {
    return this->data() >= scalar;
}
