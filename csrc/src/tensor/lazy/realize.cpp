#include "lamppp/tensor/lazy/realize.hpp"

#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/lazy/lazy_backend.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor {

void realize(TensorImpl* impl) {
  if (!impl->is_deferred())
    return;
  LazyBackend* b = backend(impl->device());
  LMP_CHECK(b) << "no lazy backend registered for device";
  b->realize(impl);
}

}  // namespace lmp::tensor
