#include "lamppp/inductor/nvrtc_backend.hpp"

#include <cstdlib>

#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/lazy/lazy_backend.hpp"
#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/lazy/realize.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::inductor {

void NVRTCInductorBackend::realize(tensor::TensorImpl* impl) {
  // §4b: walk operands then run_eager bridge; Part 2 will replace/extend this
  // body with fusion-group partitioning + NVRTC codegen.
  if (!impl->is_deferred())
    return;
  tensor::LazyFunction* fn = impl->lazy_op().get();
  for (const std::shared_ptr<tensor::TensorImpl>& in : fn->inputs) {
    if (in->is_deferred())
      tensor::realize(in.get());
  }
  fn->run_eager(*impl);
}

namespace {

/// @brief Meyers-singleton instance of the inductor backend.
NVRTCInductorBackend& instance() {
  static NVRTCInductorBackend backend;
  return backend;
}

/**
 * @brief Static-init registrar wiring the inductor backend into the tensor seam.
 *
 * @details Registration is gated on the `LMP_ENABLE_FUSION` env var: when unset
 * the backend is never registered, so `tensor::backend(...)` stays null and all
 * ops take the eager path (behavior identical to a build without inductor). The
 * gate is critical because the op-shim defer logic flips the instant any backend
 * is registered.
 */
const bool kRegistered = [] {
#ifdef LMP_ENABLE_CUDA
  if (std::getenv("LMP_ENABLE_FUSION") != nullptr) {
    tensor::register_backend(tensor::DeviceType::CUDA, &instance());
    return true;
  }
#endif
  return false;
}();

}  // namespace

}  // namespace lmp::inductor
