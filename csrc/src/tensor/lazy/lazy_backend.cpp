#include "lamppp/tensor/lazy/lazy_backend.hpp"

namespace lmp::tensor {

namespace {

std::array<LazyBackend*, static_cast<size_t>(DeviceType::Count)>& registry() {
  static std::array<LazyBackend*, static_cast<size_t>(DeviceType::Count)>
      table{};
  return table;
}

}

LazyBackend* backend(DeviceType dev) {
  return registry()[static_cast<size_t>(dev)];
}

void register_backend(DeviceType dev, LazyBackend* b) {
  registry()[static_cast<size_t>(dev)] = b;
}

}
