#ifndef DUMMY_MEM_POOL_H
#define DUMMY_MEM_POOL_H

#include <memory>
#include "autograd/engine/value_pool.h"

class FloatWrapper {
 private:
  float value_;
  static ValueMemoryPool pool_;

 public:
  explicit FloatWrapper(float value = 0.0F);
  FloatWrapper(const FloatWrapper& other);
  
  static std::shared_ptr<FloatWrapper> create(const FloatWrapper& value);
  static void destroy(FloatWrapper* ptr);

  float get() const;
  void set(float value);

  FloatWrapper operator+(const FloatWrapper& other) const;
  FloatWrapper operator-(const FloatWrapper& other) const;
  FloatWrapper operator*(const FloatWrapper& other) const;
  FloatWrapper operator/(const FloatWrapper& other) const;

  FloatWrapper& operator+=(const FloatWrapper& other);
  FloatWrapper& operator-=(const FloatWrapper& other);
  FloatWrapper& operator*=(const FloatWrapper& other);
  FloatWrapper& operator/=(const FloatWrapper& other);

  bool operator==(const FloatWrapper& other) const;
  bool operator!=(const FloatWrapper& other) const;
  bool operator<(const FloatWrapper& other) const;
  bool operator<=(const FloatWrapper& other) const;
  bool operator>(const FloatWrapper& other) const;
  bool operator>=(const FloatWrapper& other) const;
};

class SharedFloat {
public:
    SharedFloat();
    explicit SharedFloat(float value);
    explicit SharedFloat(std::shared_ptr<FloatWrapper> ptr);
    
    float getValue() const;
    void setValue(float value);
    std::shared_ptr<FloatWrapper> getPtr() const;
    
    SharedFloat operator+(const SharedFloat& other) const;
    SharedFloat operator-(const SharedFloat& other) const;
    SharedFloat operator*(const SharedFloat& other) const;
    SharedFloat operator/(const SharedFloat& other) const;
    
    SharedFloat& operator+=(const SharedFloat& other);
    SharedFloat& operator-=(const SharedFloat& other);
    SharedFloat& operator*=(const SharedFloat& other);
    SharedFloat& operator/=(const SharedFloat& other);
    
    bool operator==(const SharedFloat& other) const;
    bool operator!=(const SharedFloat& other) const;
    bool operator<(const SharedFloat& other) const;
    bool operator<=(const SharedFloat& other) const;
    bool operator>(const SharedFloat& other) const;
    bool operator>=(const SharedFloat& other) const;

private:
    std::shared_ptr<FloatWrapper> ptr_;
};

#endif // DUMMY_MEM_POOL_H
