#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lamppp/common/assert.hpp"
#include "parameter.hpp"

namespace lmp::nets {

class ModuleImpl {
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
class ModuleImplBase : public ModuleImpl {
public:
    template<typename... Args>
    autograd::Variable operator()(Args&&... args) {
        return static_cast<Derived*>(this)->forward(std::forward<Args>(args)...);
    }
};

class Module {
public:
    Module() = default;

private:
    std::shared_ptr<ModuleImpl> impl_;
};

}