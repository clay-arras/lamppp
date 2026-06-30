#pragma once

#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/lazy/lazy_backend.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor {

void realize(TensorImpl* impl);

}
