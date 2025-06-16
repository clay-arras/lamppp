#include "lamppp/nets/module.hpp"

namespace lmp::nets {

std::vector<Parameter> ModuleImpl::parameters() {
    std::vector<Parameter> all_params(params_.size());
    for (const auto& [k, v] : params_) {
        all_params.emplace_back(v);
    }
    for (auto& [k, module] : modules_) {
        auto child_params = module->parameters(); // recursive call
        all_params.insert(all_params.end(), child_params.begin(), child_params.end());
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

}