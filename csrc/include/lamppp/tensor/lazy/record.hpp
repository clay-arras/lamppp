#pragma once
#include <memory>

namespace lmp::tensor {
class TensorImpl;
class LazyFunction;

/// @brief Capture a pending op: build its deferred output impl and attach the op.
std::shared_ptr<TensorImpl> record(std::shared_ptr<LazyFunction> fn);

}  // namespace lmp::tensor
