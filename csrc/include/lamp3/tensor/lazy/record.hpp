#pragma once
#include <memory>

#include "lamp3/tensor/lazy/lazy_function.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor {

std::shared_ptr<TensorImpl> record(std::shared_ptr<LazyFunction> fn);

}
