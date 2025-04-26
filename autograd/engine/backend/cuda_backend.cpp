#include "cuda_backend.h"
#include <cassert>
#include <vector>
#include "autograd/cuda/basic_kern.cuh"
#include "autograd/cuda/unary_kern.cuh"
#include "autograd/cuda/binary_kern.cuh"
#include "autograd/cuda/matrix_kern.cuh"
#include "autograd/cuda/reduct_kern.cuh"

namespace autograd {

TensorImpl CudaBackend::add(const TensorImpl& a, const TensorImpl& b) {
  assert(a.data.size() == b.data.size());
  std::vector<float> c(a.data.size());
  vecAdd(a.data.size(), a.data.data(), b.data.data(), c.data());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::sub(const TensorImpl& a, const TensorImpl& b) {
  assert(a.data.size() == b.data.size());
  std::vector<float> c(a.data.size());
  vecSub(a.data.size(), a.data.data(), b.data.data(), c.data());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::mul(const TensorImpl& a, const TensorImpl& b) {
  assert(a.data.size() == b.data.size());
  std::vector<float> c(a.data.size());
  vecMul(a.data.size(), a.data.data(), b.data.data(), c.data());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::div(const TensorImpl& a, const TensorImpl& b) {
  assert(a.data.size() == b.data.size());
  std::vector<float> c(a.data.size());
  vecDiv(a.data.size(), a.data.data(), b.data.data(), c.data());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::log(const TensorImpl& a) {
  std::vector<float> c(a.data.size());
  vecLog(a.data.size(), a.data.data(), c.data());
  return TensorImpl(c, a.shape);
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::exp(const TensorImpl& a) {
  std::vector<float> c(a.data.size());
  vecExp(a.data.size(), a.data.data(), c.data());
  return TensorImpl(c, a.shape);
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::relu(const TensorImpl& a) {
  std::vector<float> c(a.data.size());
  vecRelu(a.data.size(), a.data.data(), c.data());
  return TensorImpl(c, a.shape);
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::matmul(const TensorImpl& a, const TensorImpl& b) {
  assert(a.shape.size() == 2 && b.shape.size() == 2);
  assert(a.shape[1] == b.shape[0]);
  
  int m = a.shape[0];
  int n = b.shape[1];
  int k = a.shape[1];
  
  std::vector<float> c(m * n);
  cudaMatMul(a.data.data(), b.data.data(), c.data(), m, n, k);
  return TensorImpl(c, {m, n});
}

TensorImpl CudaBackend::transpose(const TensorImpl& a) {
  assert(a.shape.size() == 2);
  int m = a.shape[0];
  int n = a.shape[1];
  std::vector<float> c(m * n);
  cudaTranspose(a.data.data(), c.data(), m, n);
  return TensorImpl(c, {n, m});
}

TensorImpl CudaBackend::equal(const TensorImpl& a, const TensorImpl& b) {
  assert(a.data.size() == b.data.size());
  std::vector<float> c(a.data.size());
  vecEqual(a.data.size(), a.data.data(), b.data.data(), c.data());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::not_equal(const TensorImpl& a, const TensorImpl& b) {
  assert(a.data.size() == b.data.size());
  std::vector<float> c(a.data.size());
  vecNotEqual(a.data.size(), a.data.data(), b.data.data(), c.data());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::greater_equal(const TensorImpl& a, const TensorImpl& b) {
  assert(a.data.size() == b.data.size());
  std::vector<float> c(a.data.size());
  vecGreaterEqual(a.data.size(), a.data.data(), b.data.data(), c.data());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::less_equal(const TensorImpl& a, const TensorImpl& b) {
  assert(a.data.size() == b.data.size());
  std::vector<float> c(a.data.size());
  vecLessEqual(a.data.size(), a.data.data(), b.data.data(), c.data());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::greater_than(const TensorImpl& a, const TensorImpl& b) {
  assert(a.data.size() == b.data.size());
  std::vector<float> c(a.data.size());
  vecGreaterThan(a.data.size(), a.data.data(), b.data.data(), c.data());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::less_than(const TensorImpl& a, const TensorImpl& b) {
  assert(a.data.size() == b.data.size());
  std::vector<float> c(a.data.size());
  vecLessThan(a.data.size(), a.data.data(), b.data.data(), c.data());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::sum(const TensorImpl& a, int axis) {
  assert(a.data.size() > 0);
  std::vector<float> c(a.data.size() / a.shape[axis]);
  vecSum(a.data.data(), c.data(), a.shape.data(), axis, a.shape.size());
  return TensorImpl(c, a.shape);
}

TensorImpl CudaBackend::max(const TensorImpl& a, int axis) {
  assert(a.data.size() > 0);
  std::vector<float> c(a.data.size() / a.shape[axis]);
  vecMax(a.data.data(), c.data(), a.shape.data(), axis, a.shape.size());
  return TensorImpl(c, a.shape);
}

}  // namespace autograd