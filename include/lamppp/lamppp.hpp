#pragma once

#include "autograd/core.hpp"
#include "tensor/core.hpp"
#include "nets/core.hpp"

namespace lmp {
    using Tensor = tensor::Tensor;
    using DataType = tensor::DataType;
    using DeviceType = tensor::DeviceType;
    using namespace tensor::ops;

    using Variable = autograd::Variable;
    using namespace autograd::ops;
}