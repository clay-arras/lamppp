#pragma once

#ifndef BACKEND_H
#define BACKEND_H

#include <memory>

namespace autograd {

struct Storage;

struct AbstractBackend {
  virtual Storage add(const Storage& a, const Storage& b) = 0;
  virtual Storage sub(const Storage& a, const Storage& b) = 0;
  virtual Storage mul(const Storage& a, const Storage& b) = 0;
  virtual Storage div(const Storage& a, const Storage& b) = 0;

  virtual Storage log(const Storage& a) = 0;
  virtual Storage exp(const Storage& a) = 0;
  virtual Storage relu(const Storage& a) = 0;

  virtual Storage matmul(const Storage& a, const Storage& b) = 0;
  virtual Storage transpose(const Storage& a) = 0;

  virtual Storage equal(const Storage& a, const Storage& b) = 0;
  virtual Storage not_equal(const Storage& a, const Storage& b) = 0;
  virtual Storage greater_equal(const Storage& a, const Storage& b) = 0;
  virtual Storage less_equal(const Storage& a, const Storage& b) = 0;
  virtual Storage greater_than(const Storage& a, const Storage& b) = 0;
  virtual Storage less_than(const Storage& a, const Storage& b) = 0;

  virtual Storage sum(const Storage& a, size_t axis) = 0;
  virtual Storage max(const Storage& a, size_t axis) = 0;
};

// struct
//     AbstractBackend {  // TODO(nlin): consider compile-time polymorphism, this not being static is costing ~10ns
//   virtual std::shared_ptr<TensorImpl> add(const TensorImpl& a,
//                                           const TensorImpl& b) = 0;
//   virtual std::shared_ptr<TensorImpl> sub(const TensorImpl& a,
//                                           const TensorImpl& b) = 0;
//   virtual std::shared_ptr<TensorImpl> mul(const TensorImpl& a,
//                                           const TensorImpl& b) = 0;
//   virtual std::shared_ptr<TensorImpl> div(const TensorImpl& a,
//                                           const TensorImpl& b) = 0;

//   virtual std::shared_ptr<TensorImpl> log(const TensorImpl& a) = 0;
//   virtual std::shared_ptr<TensorImpl> exp(const TensorImpl& a) = 0;
//   virtual std::shared_ptr<TensorImpl> relu(const TensorImpl& a) = 0;

//   virtual std::shared_ptr<TensorImpl> matmul(const TensorImpl& a,
//                                              const TensorImpl& b) = 0;
//   virtual std::shared_ptr<TensorImpl> transpose(const TensorImpl& a) = 0;

//   virtual std::shared_ptr<TensorImpl> equal(const TensorImpl& a,
//                                             const TensorImpl& b) = 0;
//   virtual std::shared_ptr<TensorImpl> not_equal(const TensorImpl& a,
//                                                 const TensorImpl& b) = 0;
//   virtual std::shared_ptr<TensorImpl> greater_equal(const TensorImpl& a,
//                                                     const TensorImpl& b) = 0;
//   virtual std::shared_ptr<TensorImpl> less_equal(const TensorImpl& a,
//                                                  const TensorImpl& b) = 0;
//   virtual std::shared_ptr<TensorImpl> greater_than(const TensorImpl& a,
//                                                    const TensorImpl& b) = 0;
//   virtual std::shared_ptr<TensorImpl> less_than(const TensorImpl& a,
//                                                 const TensorImpl& b) = 0;

//   virtual std::shared_ptr<TensorImpl> sum(const TensorImpl& a, size_t axis) = 0;
//   virtual std::shared_ptr<TensorImpl> max(const TensorImpl& a, size_t axis) = 0;

//   virtual ~AbstractBackend() = default;
// };

}  // namespace autograd

#endif  // BACKEND_H