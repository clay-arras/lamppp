#pragma once

#include <array>

#include "lamp3/tensor/device_type.hpp"

namespace lmp::tensor {

class TensorImpl;

namespace lazy {

class LazyFunction;

struct LazyBackend {
  virtual ~LazyBackend() = default;
  virtual void realize(TensorImpl*) = 0;
};

LazyBackend* backend(DeviceType dev);

void register_backend(DeviceType dev, LazyBackend* b);

}  // namespace lazy
}  // namespace lmp::tensor

#define LMP_REGISTER_LAZY_BACKEND(dev, backend_type)        \
  namespace {                                               \
  struct _RegLazy_##backend_type {                          \
    _RegLazy_##backend_type() {                             \
      static backend_type instance;                         \
      ::lmp::tensor::lazy::register_backend((dev), &instance); \
    }                                                        \
  } _auto_reg_lazy_##backend_type;                          \
  }
