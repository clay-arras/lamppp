#pragma once

#include <cstddef>
#include "autograd/engine/data_type.hpp"

namespace autograd {

class TensorImpl;

struct AbstractBackend {
  virtual DataType dtype_promotion_(DataType a_type, DataType b_type) = 0;

  virtual TensorImpl add(const TensorImpl& a, const TensorImpl& b) = 0;
  virtual TensorImpl sub(const TensorImpl& a, const TensorImpl& b) = 0;
  virtual TensorImpl mul(const TensorImpl& a, const TensorImpl& b) = 0;
  virtual TensorImpl div(const TensorImpl& a, const TensorImpl& b) = 0;

  virtual TensorImpl log(const TensorImpl& a) = 0;
  virtual TensorImpl exp(const TensorImpl& a) = 0;
  virtual TensorImpl relu(const TensorImpl& a) = 0;

  virtual TensorImpl matmul(const TensorImpl& a, const TensorImpl& b) = 0;
  virtual TensorImpl transpose(const TensorImpl& a) = 0;

  virtual TensorImpl equal(const TensorImpl& a, const TensorImpl& b) = 0;
  virtual TensorImpl not_equal(const TensorImpl& a, const TensorImpl& b) = 0;
  virtual TensorImpl greater_equal(const TensorImpl& a,
                                   const TensorImpl& b) = 0;
  virtual TensorImpl less_equal(const TensorImpl& a, const TensorImpl& b) = 0;
  virtual TensorImpl greater(const TensorImpl& a, const TensorImpl& b) = 0;
  virtual TensorImpl less(const TensorImpl& a, const TensorImpl& b) = 0;

  virtual TensorImpl sum(const TensorImpl& a, size_t axis) = 0;
  virtual TensorImpl max(const TensorImpl& a, size_t axis) = 0;
};

template <typename Derived>
class Singleton {
 public:
  static Derived& instance() {
    static Derived instance;
    return instance;
  }

  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;
  Singleton(Singleton&&) = delete;
  Singleton& operator=(Singleton&&) = delete;

 protected:
  Singleton() = default;
  ~Singleton() = default;
};

}  // namespace autograd
