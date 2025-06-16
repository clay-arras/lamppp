#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "parameter.hpp"

namespace lmp::nets {

class ModuleImpl {
public:
    std::vector<Parameter> parameters();
    void eval();
    void train();

protected:
    template<typename T>
    T& register_parameter(const std::string& name, T&& param) {
        params_[name] = std::forward<T>(param);
        return params_[name];
    }
    template<typename T>
    T& register_module(const std::string& name, T&& module) {
        modules_[name] = std::make_unique<T>(std::forward<T>(module));
        return *static_cast<T*>(modules_[name].get());
    }

    bool trainable_ = true;
    std::unordered_map<std::string, std::unique_ptr<ModuleImpl>> modules_;
    std::unordered_map<std::string, Parameter> params_;
};

template <typename Derived>
class Module {
public:
    Module() = default;

    std::vector<Parameter> parameters() { return impl_->parameters(); }
    void eval() { impl_->eval(); }
    void train() { impl_->train(); }

    template<typename... Args>
    autograd::Variable operator()(Args&&... args) {
        return static_cast<Derived*>(impl_.get())->forward(std::forward<Args>(args)...);
    }

protected:
    std::shared_ptr<ModuleImpl> impl_;
};

}