#pragma once

#include "autograd/engine/allocator/cuda_allocator.cuh"
#include "autograd/engine/device_type.hpp"
#ifndef _CUDA_BACKEND_H_
#define _CUDA_BACKEND_H_

#include <cassert>
#include "autograd/engine/abstract_backend.hpp"
#include "autograd/engine/storage.hpp"

namespace autograd {  // need to refactor asap

struct CudaBackend : public AbstractBackend, public Singleton<CudaBackend> {
  friend class Singleton<CudaBackend>;

  Storage add(const Storage& a, const Storage& b) override;
  Storage sub(const Storage& a, const Storage& b) override;
  Storage mul(const Storage& a, const Storage& b) override;
  Storage div(const Storage& a, const Storage& b) override;

  Storage log(const Storage& a) override;
  Storage exp(const Storage& a) override;
  Storage relu(const Storage& a) override;

  Storage matmul(const Storage& a, const Storage& b) override;
  Storage transpose(const Storage& a) override;

  Storage equal(const Storage& a, const Storage& b) override;
  Storage not_equal(const Storage& a, const Storage& b) override;
  Storage greater_equal(const Storage& a, const Storage& b) override;
  Storage less_equal(const Storage& a, const Storage& b) override;
  Storage greater(const Storage& a, const Storage& b) override;
  Storage less(const Storage& a, const Storage& b) override;

  Storage sum(const Storage& a, size_t axis) override;
  Storage max(const Storage& a, size_t axis) override;
};

}  // namespace autograd

#endif  // _CUDA_BACKEND_H_