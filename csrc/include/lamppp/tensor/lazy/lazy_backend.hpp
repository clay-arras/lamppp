#pragma once

#include "lamppp/tensor/device_type.hpp"

namespace lmp::tensor {

class TensorImpl;
class LazyFunction;

/**
 * @brief Seam between the tensor module and a lazy (graph-capturing) backend.
 *
 * @details This abstract interface lets the `tensor` module hand off realization
 * of a lazily-captured graph without depending on the `inductor` module. A
 * concrete backend is registered per device via `register_backend`, and looked
 * up at runtime via `backend`. The table is keyed by `DeviceType`, mirroring the
 * dispatch pattern used elsewhere in the module.
 *
 * @see DeviceType
 */
struct LazyBackend {
  virtual ~LazyBackend() = default;
  virtual void realize(TensorImpl*) = 0;
};

/// @brief Returns the backend registered for `dev`, or nullptr if none.
LazyBackend* backend(DeviceType dev);

/// @brief Registers `b` as the backend for `dev`, filling its slot.
void register_backend(DeviceType dev, LazyBackend* b);

}  // namespace lmp::tensor
