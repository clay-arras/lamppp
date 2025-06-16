#pragma once

#include "lamppp/autograd/constructor.hpp"
#include "lamppp/autograd/functions/matrix_ops.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/nets/module.hpp"
#include "lamppp/nets/parameter.hpp"

namespace lmp::nets {

class LinearImpl : public ModuleImplBase<LinearImpl> {
    LinearImpl(size_t in_features, size_t out_features, bool bias = true) 
        : requires_bias_(bias), weights_(autograd::randn(0, 1, {in_features, out_features}, tensor::DeviceType device, tensor::DataType dtype, bool requires_grad)) {};

    autograd::Variable forward(const autograd::Variable& x) {
        return autograd::ops::matmul(weights_, x) + static_cast<autograd::Variable>(bias_);
    }

private:
    Parameter weights_; // in x out
    Parameter bias_; // out
    bool requires_bias_;
};

}