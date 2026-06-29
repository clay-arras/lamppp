#include "lamppp/tensor/lazy/lazy_backend.hpp"

#include <array>

namespace lmp::tensor {

namespace {

/// @brief Meyers-singleton table holding one backend pointer per device.
std::array<LazyBackend*, static_cast<size_t>(DeviceType::Count)>& registry() {
  static std::array<LazyBackend*, static_cast<size_t>(DeviceType::Count)>
      table{};
  return table;
}

}  // namespace

LazyBackend* backend(DeviceType dev) {
  return registry()[static_cast<size_t>(dev)];
}

void register_backend(DeviceType dev, LazyBackend* b) {
  registry()[static_cast<size_t>(dev)] = b;
}

}  // namespace lmp::tensor
