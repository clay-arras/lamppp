#pragma once

#include "abstract_backend.hpp"
#include "dispatch_stub.hpp"

namespace autograd {

using backend_fn = AbstractBackend& (*)();
DECLARE_DISPATCH(backend_fn, backend_stub);

AbstractBackend& backend_cpu();
AbstractBackend& backend_cuda();

REGISTER_DISPATCH(backend_stub, DeviceType::CPU, backend_cpu);
REGISTER_DISPATCH(backend_stub, DeviceType::CUDA, backend_cuda);

}  // namespace autograd