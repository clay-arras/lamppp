#include "lamppp/nets/module.hpp"

namespace lmp::nets {

std::vector<Parameter> ModuleImpl::parameters() {
    std::vector<Parameter> all_params; 
    all_params.reserve(params_.size());
    for (const auto& [k, v] : params_) {
        all_params.emplace_back(v);
    }
    for (const auto& [k, module] : modules_) {
        auto child_params = module->parameters(); // recursive call
        all_params.insert(all_params.end(), child_params.begin(), child_params.end());
    }
    return all_params;
}

std::multimap<std::string, Parameter> ModuleImpl::named_parameters() {
    std::multimap<std::string, Parameter> all_params; 
    for (const auto& [k, v] : params_) {
        all_params.insert({k, v});
    }
    for (const auto& [k, module] : modules_) {
        auto child_params = module->named_parameters(); // recursive call
        all_params.insert(child_params.begin(), child_params.end());
    }
    return all_params;
}



void ModuleImpl::eval() {
    trainable_ = false; 
    for (auto& [k, module] : modules_) {
        module->eval();
    }
}

void ModuleImpl::train() {
    trainable_ = true;
    for (auto& [k, module] : modules_) {
        module->train();
    }
}

void ModuleImpl::register_parameter(const std::string& name, Parameter param) {
params_[name] = std::move(param);
}

void ModuleImpl::register_module(const std::string& name, std::shared_ptr<ModuleImpl> module) {
modules_[name] = std::move(module);
}

}