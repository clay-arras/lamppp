#pragma once

#include <array>

#include "lamppp/tensor/device_type.hpp"

namespace lmp::tensor {

class TensorImpl;
class LazyFunction;

struct LazyBackend {
  virtual ~LazyBackend() = default;
  virtual void realize(TensorImpl*) = 0;
};

LazyBackend* backend(DeviceType dev);

void register_backend(DeviceType dev, LazyBackend* b);

}

#define LMP_REGISTER_LAZY_BACKEND(dev, backend_type)        \
  namespace {                                               \
  struct _RegLazy_##backend_type {                          \
    _RegLazy_##backend_type() {                             \
      static backend_type instance;                         \
      ::lmp::tensor::register_backend((dev), &instance);    \
    }                                                        \
  } _auto_reg_lazy_##backend_type;                          \
  }
