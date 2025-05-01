#pragma once

#ifndef _CUDA_BACKEND_H_
#define _CUDA_BACKEND_H_

#include <cassert>
#include <memory>
#include <vector>
#include "autograd/cuda/basic_kern.cuh"
#include "autograd/cuda/binary_kern.cuh"
#include "autograd/cuda/matrix_kern.cuh"
#include "autograd/cuda/reduct_kern.cuh"
#include "autograd/cuda/unary_kern.cuh"
#include "autograd/engine/backend.hpp"
#include "autograd/engine/tensor_impl.hpp"

namespace autograd {  // need to refactor asap

template <typename DataType>
struct CudaBackend : public virtual AbstractBackend {
  std::shared_ptr<TensorImpl> add(const TensorImpl& a,
                                  const TensorImpl& b) override;
  std::shared_ptr<TensorImpl> sub(const TensorImpl& a,
                                  const TensorImpl& b) override;
  std::shared_ptr<TensorImpl> mul(const TensorImpl& a,
                                  const TensorImpl& b) override;
  std::shared_ptr<TensorImpl> div(const TensorImpl& a,
                                  const TensorImpl& b) override;

  std::shared_ptr<TensorImpl> log(const TensorImpl& a) override;
  std::shared_ptr<TensorImpl> exp(const TensorImpl& a) override;
  std::shared_ptr<TensorImpl> relu(const TensorImpl& a) override;

  std::shared_ptr<TensorImpl> matmul(const TensorImpl& a,
                                     const TensorImpl& b) override;
  std::shared_ptr<TensorImpl> transpose(const TensorImpl& a) override;

  std::shared_ptr<TensorImpl> equal(const TensorImpl& a,
                                    const TensorImpl& b) override;
  std::shared_ptr<TensorImpl> not_equal(const TensorImpl& a,
                                        const TensorImpl& b) override;
  std::shared_ptr<TensorImpl> greater_equal(const TensorImpl& a,
                                            const TensorImpl& b) override;
  std::shared_ptr<TensorImpl> less_equal(const TensorImpl& a,
                                         const TensorImpl& b) override;
  std::shared_ptr<TensorImpl> greater_than(const TensorImpl& a,
                                           const TensorImpl& b) override;
  std::shared_ptr<TensorImpl> less_than(const TensorImpl& a,
                                        const TensorImpl& b) override;

  std::shared_ptr<TensorImpl> sum(const TensorImpl& a, size_t axis) override;
  std::shared_ptr<TensorImpl> max(const TensorImpl& a, size_t axis) override;
};  // TODO: make a factory / struct for these methods; they are TOO REPETITIVE

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::add(const TensorImpl& a,
                                                       const TensorImpl& b) {
  assert(a.data_size() == b.data_size());
  std::vector<DataType> c(a.data_size());
  vecAdd<DataType>(a.data_size(), static_cast<const DataType*>(a.data_ptr()),
                   static_cast<const DataType*>(b.data_ptr()), c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::sub(const TensorImpl& a,
                                                       const TensorImpl& b) {
  assert(a.data_size() == b.data_size());
  std::vector<DataType> c(a.data_size());
  vecSub<DataType>(a.data_size(), static_cast<const DataType*>(a.data_ptr()),
                   static_cast<const DataType*>(b.data_ptr()), c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::mul(const TensorImpl& a,
                                                       const TensorImpl& b) {
  assert(a.data_size() == b.data_size());
  std::vector<DataType> c(a.data_size());
  vecMul<DataType>(a.data_size(), static_cast<const DataType*>(a.data_ptr()),
                   static_cast<const DataType*>(b.data_ptr()), c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::div(const TensorImpl& a,
                                                       const TensorImpl& b) {
  assert(a.data_size() == b.data_size());
  std::vector<DataType> c(a.data_size());
  vecDiv<DataType>(a.data_size(), static_cast<const DataType*>(a.data_ptr()),
                   static_cast<const DataType*>(b.data_ptr()), c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::log(const TensorImpl& a) {
  std::vector<DataType> c(a.data_size());
  vecLog<DataType>(a.data_size(), static_cast<const DataType*>(a.data_ptr()),
                   c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::exp(const TensorImpl& a) {
  std::vector<DataType> c(a.data_size());
  vecExp<DataType>(a.data_size(), static_cast<const DataType*>(a.data_ptr()),
                   c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::relu(const TensorImpl& a) {
  std::vector<DataType> c(a.data_size());
  vecRelu<DataType>(a.data_size(), static_cast<const DataType*>(a.data_ptr()),
                    c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::matmul(const TensorImpl& a,
                                                          const TensorImpl& b) {
  assert(a.shape().size() == 2 && b.shape().size() == 2);
  assert(a.shape()[1] == b.shape()[0]);
  size_t m = a.shape()[0];
  size_t n = b.shape()[1];
  size_t k = a.shape()[1];
  std::vector<DataType> c(m * n);
  cudaMatMul<DataType>(static_cast<const DataType*>(a.data_ptr()),
                       static_cast<const DataType*>(b.data_ptr()), c.data(), m,
                       n, k);
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, std::vector<size_t>{m, n}));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::transpose(
    const TensorImpl& a) {
  assert(a.shape().size() == 2);
  size_t m = a.shape()[0];
  size_t n = a.shape()[1];
  std::vector<DataType> c(m * n);
  cudaTranspose<DataType>(static_cast<const DataType*>(a.data_ptr()), c.data(),
                          m, n);
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, std::vector<size_t>{n, m}));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::equal(const TensorImpl& a,
                                                         const TensorImpl& b) {
  assert(a.data_size() == b.data_size());
  std::vector<DataType> c(a.data_size());
  vecEqual<DataType>(a.data_size(), static_cast<const DataType*>(a.data_ptr()),
                     static_cast<const DataType*>(b.data_ptr()), c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::not_equal(
    const TensorImpl& a, const TensorImpl& b) {
  assert(a.data_size() == b.data_size());
  std::vector<DataType> c(a.data_size());
  vecNotEqual<DataType>(a.data_size(),
                        static_cast<const DataType*>(a.data_ptr()),
                        static_cast<const DataType*>(b.data_ptr()), c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::greater_equal(
    const TensorImpl& a, const TensorImpl& b) {
  assert(a.data_size() == b.data_size());
  std::vector<DataType> c(a.data_size());
  vecGreaterEqual<DataType>(
      a.data_size(), static_cast<const DataType*>(a.data_ptr()),
      static_cast<const DataType*>(b.data_ptr()), c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::less_equal(
    const TensorImpl& a, const TensorImpl& b) {
  assert(a.data_size() == b.data_size());
  std::vector<DataType> c(a.data_size());
  vecLessEqual<DataType>(a.data_size(),
                         static_cast<const DataType*>(a.data_ptr()),
                         static_cast<const DataType*>(b.data_ptr()), c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::greater_than(
    const TensorImpl& a, const TensorImpl& b) {
  assert(a.data_size() == b.data_size());
  std::vector<DataType> c(a.data_size());
  vecGreaterThan<DataType>(
      a.data_size(), static_cast<const DataType*>(a.data_ptr()),
      static_cast<const DataType*>(b.data_ptr()), c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::less_than(
    const TensorImpl& a, const TensorImpl& b) {
  assert(a.data_size() == b.data_size());
  std::vector<DataType> c(a.data_size());
  vecLessThan<DataType>(a.data_size(),
                        static_cast<const DataType*>(a.data_ptr()),
                        static_cast<const DataType*>(b.data_ptr()), c.data());
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, a.shape()));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::sum(const TensorImpl& a,
                                                       size_t axis) {
  assert(a.data_size() > 0);
  std::vector<DataType> c(a.data_size() / a.shape()[axis]);
  vecSum<DataType>(static_cast<const DataType*>(a.data_ptr()), c.data(),
                   a.shape().data(), axis, a.shape().size());
  std::vector<size_t> new_shape(a.shape().begin(), a.shape().end());
  new_shape[axis] = 1;
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, new_shape));
}

template <typename DataType>
std::shared_ptr<TensorImpl> CudaBackend<DataType>::max(const TensorImpl& a,
                                                       size_t axis) {
  assert(a.data_size() > 0);
  std::vector<DataType> c(a.data_size() / a.shape()[axis]);
  vecMax<DataType>(static_cast<const DataType*>(a.data_ptr()), c.data(),
                   a.shape().data(), axis, a.shape().size());
  std::vector<size_t> new_shape(a.shape().begin(), a.shape().end());
  new_shape[axis] = 1;
  return std::static_pointer_cast<TensorImpl>(
      std::make_shared<TensorImplModel<DataType, CudaBackend<DataType>>>(
          c, new_shape));
}

}  // namespace autograd

#endif  // _CUDA_BACKEND_H_