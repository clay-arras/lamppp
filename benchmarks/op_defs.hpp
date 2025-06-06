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

class BinaryOperatorBase : public OperatorBase {
public:
    void register_benchmarks(const OperatorConfig& config) override {
        auto op_fn = [this](const std::array<lmp::autograd::Variable, 2>& inputs) -> lmp::autograd::Variable {
            return apply_operation(inputs[0], inputs[1]);
        };
        auto init_fn = [config]() -> std::array<lmp::autograd::Variable, 2> {
            return {
                lmp::autograd::rand(config.shape, config.device, config.dtype, true),
                lmp::autograd::rand(config.shape, config.device, config.dtype, true)
            };
        };
        
        std::string bench_name = name() + "_" + 
                                std::to_string(config.shape[0]) + "x" + 
                                std::to_string(config.shape[1]) + "_" +
                                (config.device == lmp::tensor::DeviceType::CUDA ? "CUDA" : "CPU");
        
        register_forward<2>(bench_name, op_fn, init_fn);
        register_backward<2>(bench_name, op_fn, init_fn);
    }

protected:
    virtual lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a, const lmp::autograd::Variable& b) = 0;
};

class UnaryOperatorBase : public OperatorBase {
public:
    void register_benchmarks(const OperatorConfig& config) override {
        auto op_fn = [this](const std::array<lmp::autograd::Variable, 1>& inputs) -> lmp::autograd::Variable {
            return apply_operation(inputs[0]);
        };
        auto init_fn = [config]() -> std::array<lmp::autograd::Variable, 1> {
            return {lmp::autograd::rand(config.shape, config.device, config.dtype, true)};
        };
        
        std::string bench_name = name() + "_" + 
                                std::to_string(config.shape[0]) + "x" + 
                                std::to_string(config.shape[1]) + "_" +
                                (config.device == lmp::tensor::DeviceType::CUDA ? "CUDA" : "CPU");
        
        register_forward<1>(bench_name, op_fn, init_fn);
        register_backward<1>(bench_name, op_fn, init_fn);
    }

protected:
    virtual lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a) = 0;
};

class AddOp : public BinaryOperatorBase {
public:
    std::string name() const override { return "Add"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a, const lmp::autograd::Variable& b) override {
        return a + b;
    }
};

class SubOp : public BinaryOperatorBase {
public:
    std::string name() const override { return "Sub"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a, const lmp::autograd::Variable& b) override {
        return a - b;
    }
};

class MulOp : public BinaryOperatorBase {
public:
    std::string name() const override { return "Mul"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a, const lmp::autograd::Variable& b) override {
        return a * b;
    }
};

class DivOp : public BinaryOperatorBase {
public:
    std::string name() const override { return "Div"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a, const lmp::autograd::Variable& b) override {
        return a / b;
    }
};

class PowOp : public BinaryOperatorBase {
public:
    std::string name() const override { return "Pow"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a, const lmp::autograd::Variable& b) override {
        return lmp::autograd::ops::pow(a, b);
    }
};

// Unary Operations
class LogOp : public UnaryOperatorBase {
public:
    std::string name() const override { return "Log"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a) override {
        return lmp::autograd::ops::log(a);
    }
};

class ExpOp : public UnaryOperatorBase {
public:
    std::string name() const override { return "Exp"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a) override {
        return lmp::autograd::ops::exp(a);
    }
};

class SqrtOp : public UnaryOperatorBase {
public:
    std::string name() const override { return "Sqrt"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a) override {
        return lmp::autograd::ops::sqrt(a);
    }
};

class AbsOp : public UnaryOperatorBase {
public:
    std::string name() const override { return "Abs"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a) override {
        return lmp::autograd::ops::abs(a);
    }
};

class SinOp : public UnaryOperatorBase {
public:
    std::string name() const override { return "Sin"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a) override {
        return lmp::autograd::ops::sin(a);
    }
};

class CosOp : public UnaryOperatorBase {
public:
    std::string name() const override { return "Cos"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a) override {
        return lmp::autograd::ops::cos(a);
    }
};

class TanOp : public UnaryOperatorBase {
public:
    std::string name() const override { return "Tan"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a) override {
        return lmp::autograd::ops::tan(a);
    }
};

class ClampOp : public UnaryOperatorBase {
public:
    std::string name() const override { return "Clamp"; }
protected:
    lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a) override {
        return lmp::autograd::ops::clamp(a, 0.25F, 0.75F);
    }
}; 