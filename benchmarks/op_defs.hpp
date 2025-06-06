#pragma once

#include <vector>
#include <string>
#include <memory>
#include "lamppp/lamppp.hpp"
#include "reg_defs.hpp"

struct OperatorConfig {
    std::vector<size_t> shape;
    lmp::tensor::DeviceType device;
    lmp::tensor::DataType dtype;
};

class OperatorBase {
public:
    virtual ~OperatorBase() = default;
    virtual void register_benchmarks(const OperatorConfig& config) = 0;
    virtual std::string name() const = 0;
};

class AddOp : public OperatorBase {
public:
    std::string name() const override { return "Add"; }
    
    void register_benchmarks(const OperatorConfig& config) override {
        auto op_fn = [](const std::array<lmp::autograd::Variable, 2>& inputs) -> lmp::autograd::Variable {
            return inputs[0] + inputs[1];
        };
        auto init_fn_grad = [config]() -> std::array<lmp::autograd::Variable, 2> {
            return {
                lmp::autograd::rand(config.shape, config.device, config.dtype, true),
                lmp::autograd::rand(config.shape, config.device, config.dtype, true)
            };
        };
        
        std::string bench_name = name() + "_" + 
                                std::to_string(config.shape[0]) + "x" + 
                                std::to_string(config.shape[1]) + "_" +
                                (config.device == lmp::tensor::DeviceType::CUDA ? "CUDA" : "CPU");
        
        register_forward<2>(bench_name, op_fn, init_fn_grad);
        register_backward<2>(bench_name, op_fn, init_fn_grad);
    }
}; 