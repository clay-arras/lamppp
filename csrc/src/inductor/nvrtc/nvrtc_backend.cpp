#include "lamppp/inductor/nvrtc/nvrtc_backend.hpp"

#include "lamppp/common/assert.hpp"
#include "lamppp/inductor/nvrtc/fused_graph.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/lazy/lazy_backend.hpp"
#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::inductor {

void NVRTCInductorBackend::realize(tensor::TensorImpl* impl) {
  if (!impl->is_deferred())
    return;
  FusedGraph g =
      build_fused_graph(impl);  // forces all boundary inputs as a side effect
  // v0: execute interior nodes eagerly in evaluation order.
  // NOTE(next piece): replace this loop with codegen of ONE fused kernel from
  // g.order + g.inputs/slot (via LazyFunction::codegen_expr) and a single launch.
  for (tensor::TensorImpl* n : g.order)
    n->lazy_op()->run_eager(*n);
}

// Master switch for kernel fusion. Three separate concerns, kept distinct:
//   - LMP_ENABLE_CUDA  : whether this TU (the NVRTC backend) is compiled at all
//   - LMP_ENABLE_FUSION: whether the lazy backend registers (this gate)
//   - backend(device)  : the op-shim's per-op defer decision (reads registration)
// When LMP_ENABLE_FUSION is off, no backend registers, backend(CUDA) is null,
// and every op stays on the eager path (bit-identical to a no-inductor build).
// NOTE(future): when fusion is off we should go further and not build/link the
// inductor library at all, rather than building it and compiling out only the
// registrar. Left as a follow-up.
#ifdef LMP_ENABLE_FUSION
LMP_REGISTER_LAZY_BACKEND(tensor::DeviceType::CUDA, NVRTCInductorBackend)
#endif

}  // namespace lmp::inductor
