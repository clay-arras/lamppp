#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lamppp/common/assert.hpp"
#include "parameter.hpp"

class Module {
public:
    Module() = default;

private:
    class ModuleImpl;
    std::unique_ptr<ModuleImpl> impl_;
};

class Module::ModuleImpl {
public:
    std::vector<Parameter> parameters();
    void eval();
    void train();

    template<typename Ret, typename... Args>
    Ret forward(Args&&...  /*args*/) {
        LMP_INTERNAL_ASSERT(false) << "Not Implemented";
    }

    template<typename Ret, typename... Args>
    Ret operator()(Args&&... args) {
        return forward<Ret>(std::forward<Args>(args)...);
    }

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

private:
    bool trainable_ = true;
    std::unordered_map<std::string, std::unique_ptr<ModuleImpl>> modules_;
    std::unordered_map<std::string, Parameter> params_;
};