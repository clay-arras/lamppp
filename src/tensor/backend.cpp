#include "include/lamppp/tensor/backend.hpp"
#include "include/lamppp/tensor/backend/cuda_backend.hpp"
#include "include/lamppp/tensor/dispatch_stub.hpp"

namespace autograd {

DEFINE_DISPATCH(backend_stub)

AbstractBackend& backend_cpu() {
  assert(false && "Not implemented");
}

AbstractBackend& backend_cuda() {
  return CudaBackend::instance();
}
}  // namespace autograd