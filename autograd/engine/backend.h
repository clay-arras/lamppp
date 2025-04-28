#pragma once

#ifndef BACKEND_H
#define BACKEND_H

#include <memory>
#include "autograd/engine/tensor_impl.h"

namespace autograd {

struct
    AbstractBackend {  // TODO(nlin): consider compile-time polymorphism, this not being static is costing ~10ns
  virtual std::shared_ptr<TensorImpl> add(const TensorImpl& a,
                                          const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> sub(const TensorImpl& a,
                                          const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> mul(const TensorImpl& a,
                                          const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> div(const TensorImpl& a,
                                          const TensorImpl& b) = 0;

  virtual std::shared_ptr<TensorImpl> log(const TensorImpl& a) = 0;
  virtual std::shared_ptr<TensorImpl> exp(const TensorImpl& a) = 0;
  virtual std::shared_ptr<TensorImpl> relu(const TensorImpl& a) = 0;

  virtual std::shared_ptr<TensorImpl> matmul(const TensorImpl& a,
                                             const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> transpose(const TensorImpl& a) = 0;

  virtual std::shared_ptr<TensorImpl> equal(const TensorImpl& a,
                                            const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> not_equal(const TensorImpl& a,
                                                const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> greater_equal(const TensorImpl& a,
                                                    const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> less_equal(const TensorImpl& a,
                                                 const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> greater_than(const TensorImpl& a,
                                                   const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> less_than(const TensorImpl& a,
                                                const TensorImpl& b) = 0;

  virtual std::shared_ptr<TensorImpl> sum(const TensorImpl& a, int axis) = 0;
  virtual std::shared_ptr<TensorImpl> max(const TensorImpl& a, int axis) = 0;
  // virtual std::shared_ptr<TensorImpl> mean(const TensorImpl& a, int axis) = 0;
  // virtual std::shared_ptr<TensorImpl> min(const TensorImpl& a, int axis) = 0;

  virtual ~AbstractBackend() = default;
};

}  // namespace autograd

#endif  // BACKEND_H