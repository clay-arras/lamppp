#pragma once
#include <memory>

#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor {

std::shared_ptr<TensorImpl> record(std::shared_ptr<LazyFunction> fn);

}
