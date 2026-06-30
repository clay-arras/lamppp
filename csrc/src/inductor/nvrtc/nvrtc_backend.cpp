#include "lamppp/inductor/nvrtc/nvrtc_backend.hpp"

#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/lazy/lazy_backend.hpp"
#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/lazy/realize.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::inductor {

void NVRTCInductorBackend::realize(tensor::TensorImpl* impl) {
  if (!impl->is_deferred())
    return;
  tensor::LazyFunction* fn = impl->lazy_op().get();
  for (const std::shared_ptr<tensor::TensorImpl>& in : fn->inputs) {
    if (in->is_deferred())
      tensor::realize(in.get());
  }
  fn->run_eager(*impl);
}

LMP_REGISTER_LAZY_BACKEND(tensor::DeviceType::CUDA, NVRTCInductorBackend)

}  // namespace lmp::inductor
