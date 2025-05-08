#include "backend.hpp"
#include "autograd/engine/backend/cuda_backend.hpp"

namespace autograd {

AbstractBackend& backend_cpu() {
  assert(false);
}

AbstractBackend& backend_cuda() {
  return CudaBackend::instance();
}
}  // namespace autograd