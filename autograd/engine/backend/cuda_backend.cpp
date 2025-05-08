#include "cuda_backend.hpp"
#include <vector>
#include "autograd/cuda/basic_kern.cuh"
#include "autograd/cuda/binary_kern.cuh"
#include "autograd/cuda/matrix_kern.cuh"
#include "autograd/cuda/reduct_kern.cuh"
#include "autograd/cuda/unary_kern.cuh"
#include "autograd/engine/dispatch.hpp"

namespace autograd {

Storage CudaBackend::add(const Storage& a, const Storage& b) {
  assert(a.size() == b.size());
  assert(a.type() == b.type());
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecAdd<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                     static_cast<const scalar_t*>(b.data()),
                     static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::sub(const Storage& a, const Storage& b) {
  assert(a.size() == b.size());
  assert(a.type() == b.type());
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecSub<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                     static_cast<const scalar_t*>(b.data()),
                     static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::mul(const Storage& a, const Storage& b) {
  assert(a.size() == b.size());
  assert(a.type() == b.type());
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecMul<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                     static_cast<const scalar_t*>(b.data()),
                     static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::div(const Storage& a, const Storage& b) {
  assert(a.size() == b.size());
  assert(a.type() == b.type());
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecDiv<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                     static_cast<const scalar_t*>(b.data()),
                     static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::log(const Storage& a) {
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecLog<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                     static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::exp(const Storage& a) {
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecExp<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                     static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::relu(const Storage& a) {
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecRelu<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                      static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::matmul(const Storage& a, const Storage& b) {
  assert(a.shape().size() == 2 && b.shape().size() == 2);
  assert(a.shape()[1] == b.shape()[0]);
  assert(a.type() == b.type());

  size_t m = a.shape()[0];
  size_t n = b.shape()[1];
  size_t k = a.shape()[1];

  std::vector<size_t> result_shape = {m, n};
  Storage c(result_shape, a.type(), this->default_allocator());

  DISPATCH_ALL_TYPES(c.type(), [&] {
    cudaMatMul<scalar_t>(static_cast<const scalar_t*>(a.data()),
                         static_cast<const scalar_t*>(b.data()),
                         static_cast<scalar_t*>(c.data()), m, n, k);
  });
  return c;
}

Storage CudaBackend::transpose(const Storage& a) {
  assert(a.shape().size() == 2);
  size_t m = a.shape()[0];
  size_t n = a.shape()[1];

  std::vector<size_t> result_shape = {n, m};
  Storage c(result_shape, a.type(), this->default_allocator());

  DISPATCH_ALL_TYPES(c.type(), [&] {
    cudaTranspose<scalar_t>(static_cast<const scalar_t*>(a.data()),
                            static_cast<scalar_t*>(c.data()), m, n);
  });
  return c;
}

Storage CudaBackend::equal(const Storage& a, const Storage& b) {
  assert(a.size() == b.size());
  assert(a.type() == b.type());
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecEqual<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                       static_cast<const scalar_t*>(b.data()),
                       static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::not_equal(const Storage& a, const Storage& b) {
  assert(a.size() == b.size());
  assert(a.type() == b.type());
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecNotEqual<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                          static_cast<const scalar_t*>(b.data()),
                          static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::greater_equal(const Storage& a, const Storage& b) {
  assert(a.size() == b.size());
  assert(a.type() == b.type());
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecGreaterEqual<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                              static_cast<const scalar_t*>(b.data()),
                              static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::less_equal(const Storage& a, const Storage& b) {
  assert(a.size() == b.size());
  assert(a.type() == b.type());
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecLessEqual<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                           static_cast<const scalar_t*>(b.data()),
                           static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::greater(const Storage& a, const Storage& b) {
  assert(a.size() == b.size());
  assert(a.type() == b.type());
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecGreaterThan<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                             static_cast<const scalar_t*>(b.data()),
                             static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::less(const Storage& a, const Storage& b) {
  assert(a.size() == b.size());
  assert(a.type() == b.type());
  Storage c(a.shape(), a.type(), this->default_allocator());
  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecLessThan<scalar_t>(c.size(), static_cast<const scalar_t*>(a.data()),
                          static_cast<const scalar_t*>(b.data()),
                          static_cast<scalar_t*>(c.data()));
  });
  return c;
}

Storage CudaBackend::sum(const Storage& a, size_t axis) {
  assert(a.size() > 0);
  assert(axis < a.shape().size());

  std::vector<size_t> result_shape = a.shape();
  result_shape[axis] = 1;

  Storage c(result_shape, a.type(), this->default_allocator());

  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecSum<scalar_t>(static_cast<const scalar_t*>(a.data()),
                     static_cast<scalar_t*>(c.data()), a.shape().data(), axis,
                     a.shape().size());
  });

  return c;
}

Storage CudaBackend::max(const Storage& a, size_t axis) {
  assert(a.size() > 0);
  assert(axis < a.shape().size());

  std::vector<size_t> result_shape = a.shape();
  result_shape[axis] = 1;

  Storage c(result_shape, a.type(), this->default_allocator());

  DISPATCH_ALL_TYPES(c.type(), [&] {
    vecMax<scalar_t>(static_cast<const scalar_t*>(a.data()),
                     static_cast<scalar_t*>(c.data()), a.shape().data(), axis,
                     a.shape().size());
  });

  return c;
}

}  // namespace autograd