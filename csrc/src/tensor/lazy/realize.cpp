#include "lamp3/tensor/lazy/realize.hpp"

namespace lmp::tensor::lazy {

void realize(TensorImpl* impl) {
  if (!impl->is_deferred()) return;
  LazyBackend* b = backend(impl->device());
  LMP_CHECK(b) << "no lazy backend registered for device";
  b->realize(impl);
}

}  // namespace lmp::tensor::lazy
