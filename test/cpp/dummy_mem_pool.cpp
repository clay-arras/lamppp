#include <memory>
#include "dummy_mem_pool.h"

ValueMemoryPool FloatWrapper::pool_(1000, sizeof(FloatWrapper));

FloatWrapper::FloatWrapper(float value) : value_(value) {}

FloatWrapper::FloatWrapper(const FloatWrapper& other) = default;

std::shared_ptr<FloatWrapper> FloatWrapper::create(const FloatWrapper& value) {
  void* raw_memory = pool_.allocate();
  auto* block = new (raw_memory) FloatWrapper(value);

  std::shared_ptr<FloatWrapper> alloc(block, destroy);
  return alloc;
}

void FloatWrapper::destroy(FloatWrapper* ptr) {
  ptr->~FloatWrapper();
  pool_.deallocate(static_cast<void*>(ptr));
}

float FloatWrapper::get() const { return value_; }
void FloatWrapper::set(float value) { value_ = value; }

FloatWrapper FloatWrapper::operator+(const FloatWrapper& other) const {
  return FloatWrapper(value_ + other.value_);
}

FloatWrapper FloatWrapper::operator-(const FloatWrapper& other) const {
  return FloatWrapper(value_ - other.value_);
}

FloatWrapper FloatWrapper::operator*(const FloatWrapper& other) const {
  return FloatWrapper(value_ * other.value_);
}

FloatWrapper FloatWrapper::operator/(const FloatWrapper& other) const {
  return FloatWrapper(value_ / other.value_);
}

FloatWrapper& FloatWrapper::operator+=(const FloatWrapper& other) {
  value_ += other.value_;
  return *this;
}

FloatWrapper& FloatWrapper::operator-=(const FloatWrapper& other) {
  value_ -= other.value_;
  return *this;
}

FloatWrapper& FloatWrapper::operator*=(const FloatWrapper& other) {
  value_ *= other.value_;
  return *this;
}

FloatWrapper& FloatWrapper::operator/=(const FloatWrapper& other) {
  value_ /= other.value_;
  return *this;
}

bool FloatWrapper::operator==(const FloatWrapper& other) const {
  return value_ == other.value_;
}

bool FloatWrapper::operator!=(const FloatWrapper& other) const {
  return value_ != other.value_;
}

bool FloatWrapper::operator<(const FloatWrapper& other) const {
  return value_ < other.value_;
}

bool FloatWrapper::operator<=(const FloatWrapper& other) const {
  return value_ <= other.value_;
}

bool FloatWrapper::operator>(const FloatWrapper& other) const {
  return value_ > other.value_;
}

bool FloatWrapper::operator>=(const FloatWrapper& other) const {
  return value_ >= other.value_;
}

// SharedFloat implementation
SharedFloat::SharedFloat() : ptr_(FloatWrapper::create(FloatWrapper())) {}

SharedFloat::SharedFloat(float value) : ptr_(FloatWrapper::create(FloatWrapper(value))) {}

SharedFloat::SharedFloat(std::shared_ptr<FloatWrapper> ptr) : ptr_(std::move(ptr)) {}

float SharedFloat::getValue() const { return ptr_->get(); }

void SharedFloat::setValue(float value) { ptr_->set(value); }

std::shared_ptr<FloatWrapper> SharedFloat::getPtr() const { return ptr_; }

SharedFloat SharedFloat::operator+(const SharedFloat& other) const {
    return SharedFloat(FloatWrapper::create(FloatWrapper(getValue() + other.getValue())));
}

SharedFloat SharedFloat::operator-(const SharedFloat& other) const {
    return SharedFloat(FloatWrapper::create(FloatWrapper(getValue() - other.getValue())));
}

SharedFloat SharedFloat::operator*(const SharedFloat& other) const {
    return SharedFloat(FloatWrapper::create(FloatWrapper(getValue() * other.getValue())));
}

SharedFloat SharedFloat::operator/(const SharedFloat& other) const {
    return SharedFloat(FloatWrapper::create(FloatWrapper(getValue() / other.getValue())));
}

SharedFloat& SharedFloat::operator+=(const SharedFloat& other) {
    setValue(getValue() + other.getValue());
    return *this;
}

SharedFloat& SharedFloat::operator-=(const SharedFloat& other) {
    setValue(getValue() - other.getValue());
    return *this;
}

SharedFloat& SharedFloat::operator*=(const SharedFloat& other) {
    setValue(getValue() * other.getValue());
    return *this;
}

SharedFloat& SharedFloat::operator/=(const SharedFloat& other) {
    setValue(getValue() / other.getValue());
    return *this;
}

bool SharedFloat::operator==(const SharedFloat& other) const {
    return getValue() == other.getValue();
}

bool SharedFloat::operator!=(const SharedFloat& other) const {
    return getValue() != other.getValue();
}

bool SharedFloat::operator<(const SharedFloat& other) const {
    return getValue() < other.getValue();
}

bool SharedFloat::operator<=(const SharedFloat& other) const {
    return getValue() <= other.getValue();
}

bool SharedFloat::operator>(const SharedFloat& other) const {
    return getValue() > other.getValue();
}

bool SharedFloat::operator>=(const SharedFloat& other) const {
    return getValue() >= other.getValue();
}