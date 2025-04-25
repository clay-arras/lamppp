#include "cuda_backend.h"
#include <cassert>
#include <vector>
#include "autograd/cuda/basic_kern.cuh"

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

TensorImpl CudaBackend::log(const TensorImpl& /*a*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::exp(const TensorImpl& /*a*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::relu(const TensorImpl& /*a*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::matmul(const TensorImpl& /*a*/,
                               const TensorImpl& /*b*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::transpose(const TensorImpl& /*a*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::equal(const TensorImpl& /*a*/,
                              const TensorImpl& /*b*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::not_equal(const TensorImpl& /*a*/,
                                  const TensorImpl& /*b*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::greater_equal(const TensorImpl& /*a*/,
                                      const TensorImpl& /*b*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::less_equal(const TensorImpl& /*a*/,
                                   const TensorImpl& /*b*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::greater_than(const TensorImpl& /*a*/,
                                     const TensorImpl& /*b*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::less_than(const TensorImpl& /*a*/,
                                  const TensorImpl& /*b*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::sum(const TensorImpl& /*a*/, int /*axis*/) {
  return TensorImpl({}, {});
}

TensorImpl CudaBackend::max(const TensorImpl& /*a*/, int /*axis*/) {
  return TensorImpl({}, {});
}

}  // namespace autograd