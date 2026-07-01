#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <mutex>
#include <string>
#include <vector>

#include "lamp3/inductor/nvrtc/assert.hpp"
#include "lamp3/inductor/nvrtc/codegen.hpp"
#include "lamp3/inductor/nvrtc/fused_graph.hpp"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/device_type.hpp"
#include "lamp3/tensor/dispatch_type.hpp"
#include "lamp3/tensor/lazy/lazy_backend.hpp"
#include "lamp3/tensor/lazy/lazy_function.hpp"
#include "lamp3/tensor/lazy/realize.hpp"
#include "lamp3/tensor/storage.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::inductor {

class NVRTCInductorBackend : public tensor::LazyBackend {
 public:
  void realize(tensor::TensorImpl* impl) override;
};

}
