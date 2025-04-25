#pragma once

#ifndef BACKEND_H
#define BACKEND_H

#include "autograd/engine/tensor_impl.h"

namespace autograd {

struct
    AbstractBackend {  // TODO(nlin): consider compile-time polymorphism, this not being static is costing ~10ns
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
  virtual TensorImpl greater_than(const TensorImpl& a, const TensorImpl& b) = 0;
  virtual TensorImpl less_than(const TensorImpl& a, const TensorImpl& b) = 0;

  virtual TensorImpl sum(const TensorImpl& a, int axis) = 0;
  virtual TensorImpl max(const TensorImpl& a, int axis) = 0;
  // virtual TensorImpl mean(const TensorImpl& a, int axis) = 0;
  // virtual TensorImpl min(const TensorImpl& a, int axis) = 0;

  virtual ~AbstractBackend() = default;
};

}  // namespace autograd

#endif  // BACKEND_H