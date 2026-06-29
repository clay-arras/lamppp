#pragma once

#include "lamppp/tensor/lazy/lazy_backend.hpp"

namespace lmp::inductor {

/**
 * @brief NVRTC-backed concrete `tensor::LazyBackend` (scaffold).
 *
 * @details Realizes a lazily-captured graph by JIT-compiling fused CUDA kernels
 * via NVRTC. This is the single entry point the `tensor` module reaches through
 * the abstract `tensor::LazyBackend` seam, keeping the dependency direction
 * strictly one-way (`inductor` -> `tensor`).
 *
 * @note `realize` is currently an asserting stub; the actual codegen/NVRTC path
 * is implemented in a later piece (§4b).
 *
 * @see tensor::LazyBackend
 */
class NVRTCInductorBackend : public tensor::LazyBackend {
 public:
  void realize(tensor::TensorImpl* impl) override;
};

}  // namespace lmp::inductor
