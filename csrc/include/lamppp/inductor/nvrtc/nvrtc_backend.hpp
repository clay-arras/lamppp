#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <mutex>
#include <string>
#include <vector>

#include "lamppp/inductor/nvrtc/assert.hpp"
#include "lamppp/inductor/nvrtc/codegen.hpp"
#include "lamppp/inductor/nvrtc/fused_graph.hpp"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_type.hpp"
#include "lamppp/tensor/lazy/lazy_backend.hpp"
#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/lazy/realize.hpp"
#include "lamppp/tensor/storage.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::inductor {

class NVRTCInductorBackend : public tensor::LazyBackend {
 public:
  void realize(tensor::TensorImpl* impl) override;
};

}
