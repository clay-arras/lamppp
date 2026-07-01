#include "lamp3/tensor/lazy/lazy_backend.hpp"

namespace lmp::tensor::lazy {

namespace {

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

}  // namespace lmp::tensor::lazy
