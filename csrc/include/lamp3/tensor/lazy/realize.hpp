#pragma once

#include "lamp3/common/assert.hpp"
#include "lamp3/tensor/lazy/lazy_backend.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::lazy {

void realize(TensorImpl* impl);

}  // namespace lmp::tensor::lazy
