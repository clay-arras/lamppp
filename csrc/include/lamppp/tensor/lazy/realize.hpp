#pragma once

namespace lmp::tensor {

class TensorImpl;

/// @brief Force realization of a deferred tensor via its device's backend.
void realize(TensorImpl* impl);

}  // namespace lmp::tensor
