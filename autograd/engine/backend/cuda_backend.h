#pragma once

#ifndef _CUDA_BACKEND_H_
#define _CUDA_BACKEND_H_

#include "autograd/engine/backend.h"

namespace autograd {

struct CudaBackend : AbstractBackend {
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
  TensorImpl greater_than(const TensorImpl& a, const TensorImpl& b) override;
  TensorImpl less_than(const TensorImpl& a, const TensorImpl& b) override;

  TensorImpl sum(const TensorImpl& a, int axis) override;
  TensorImpl max(const TensorImpl& a, int axis) override;
  // TensorImpl mean(const TensorImpl& a, int axis) override;
  // TensorImpl min(const TensorImpl& a, int axis) override;
};

}  // namespace autograd

#endif  // _CUDA_BACKEND_H_