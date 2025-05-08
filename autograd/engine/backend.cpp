#include "backend.hpp"
#include "autograd/engine/backend/cuda_backend.hpp"
#include "autograd/engine/dispatch_stub.hpp"

namespace autograd {

DEFINE_DISPATCH(backend_stub)

AbstractBackend& backend_cpu() {
  assert(false);
}

AbstractBackend& backend_cuda() {
  return CudaBackend::instance();
}
}  // namespace autograd