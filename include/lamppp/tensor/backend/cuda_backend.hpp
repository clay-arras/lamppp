#pragma once

#include <cassert>
#include "include/lamppp/tensor/abstract_backend.hpp"

namespace autograd {

struct CudaBackend : public AbstractBackend, public Singleton<CudaBackend> {
  friend class Singleton<CudaBackend>;
  DataType dtype_promotion_(DataType a_type, DataType b_type) override;

  TensorImpl add(const TensorImpl& a, const TensorImpl& b) override;
  TensorImpl sub(const TensorImpl& a, const TensorImpl& b) override;
  TensorImpl mul(const TensorImpl& a, const TensorImpl& b) override;
  TensorImpl div(const TensorImpl& a, const TensorImpl& b) override;

  TensorImpl log(const TensorImpl& a) override;
  TensorImpl exp(const TensorImpl& a) override;
  TensorImpl relu(const TensorImpl& a) override;

  TensorImpl matmul(const TensorImpl& a, const TensorImpl& b) override;
  TensorImpl transpose(const TensorImpl& a) override;

  TensorImpl equal(const TensorImpl& a, const TensorImpl& b) override;
  TensorImpl not_equal(const TensorImpl& a, const TensorImpl& b) override;
  TensorImpl greater_equal(const TensorImpl& a, const TensorImpl& b) override;
  TensorImpl less_equal(const TensorImpl& a, const TensorImpl& b) override;
  TensorImpl greater(const TensorImpl& a, const TensorImpl& b) override;
  TensorImpl less(const TensorImpl& a, const TensorImpl& b) override;

  TensorImpl sum(const TensorImpl& a, size_t axis) override;
  TensorImpl max(const TensorImpl& a, size_t axis) override;
};

}  // namespace autograd
