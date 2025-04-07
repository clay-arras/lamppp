#include <memory>
#include "dummy_value.h"

Float::Float() : impl_(std::make_unique<Impl>()) {}

Float::Float(float value) : impl_(std::make_unique<Impl>(value)) {}

Float::Float(const Float& other) : impl_(std::make_unique<Impl>(other.impl_->value)) {}
Float& Float::operator=(const Float& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(other.impl_->value);
    }
    return *this;
}

Float::Float(Float&& other) noexcept = default;
Float& Float::operator=(Float&& other) noexcept = default;

float Float::getValue() const { return impl_->value; }
void Float::setValue(float value) { impl_->value = value; }

Float Float::operator+(const Float& other) const {
    return Float(impl_->value + other.impl_->value);
}

Float Float::operator-(const Float& other) const {
    return Float(impl_->value - other.impl_->value);
}

Float Float::operator*(const Float& other) const {
    return Float(impl_->value * other.impl_->value);
}

Float Float::operator/(const Float& other) const {
    return Float(impl_->value / other.impl_->value);
}

Float& Float::operator+=(const Float& other) {
    impl_->value += other.impl_->value;
    return *this;
}

Float& Float::operator-=(const Float& other) {
    impl_->value -= other.impl_->value;
    return *this;
}

Float& Float::operator*=(const Float& other) {
    impl_->value *= other.impl_->value;
    return *this;
}

Float& Float::operator/=(const Float& other) {
    impl_->value /= other.impl_->value;
    return *this;
}

// Comparison operators
bool Float::operator==(const Float& other) const {
    return impl_->value == other.impl_->value;
}

bool Float::operator!=(const Float& other) const {
    return impl_->value != other.impl_->value;
}

bool Float::operator<(const Float& other) const {
    return impl_->value < other.impl_->value;
}

bool Float::operator<=(const Float& other) const {
    return impl_->value <= other.impl_->value;
}

bool Float::operator>(const Float& other) const {
    return impl_->value > other.impl_->value;
}

bool Float::operator>=(const Float& other) const {
    return impl_->value >= other.impl_->value;
}