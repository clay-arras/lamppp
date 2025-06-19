#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "parameter.hpp"

namespace lmp::nets {

namespace detail {
template <typename T>
class UnsafeModuleAccessor;
}

class ModuleImpl {
 public:
  std::vector<Parameter> parameters();
  void eval();
  void train();

 protected:
  template <typename T>
  T& register_parameter(const std::string& name, T&& param) {
    params_[name] = std::forward<T>(param);
    return params_[name];
  }
  template <typename T>
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
  template <typename... Args>
  explicit Module(Args&&... args)
      : impl_(std::make_shared<Derived>(std::forward<Args>(args)...)) {}

  std::vector<Parameter> parameters();
  void eval();
  void train();

  template <typename... Args>
  auto operator()(Args&&... args)
      -> std::invoke_result_t<decltype(&Derived::forward), Derived, Args...> {
    return static_cast<Derived*>(impl_.get())
        ->forward(std::forward<Args>(args)...);
  }

 protected:
  std::shared_ptr<Derived> impl_;

  template <typename T>
  friend class detail::UnsafeModuleAccessor;
};

namespace detail {
// @internal
template <typename T>
struct UnsafeModuleAccessor {
  static std::shared_ptr<T> getImpl(const Module<T>& mod) {
    return mod.impl_;
  }
};
// @endinternal
}  // namespace detail

template <typename Derived>
std::vector<Parameter> Module<Derived>::parameters() {
  return impl_->parameters();
}

template <typename Derived>
void Module<Derived>::eval() {
  impl_->eval();
}

template <typename Derived>
void Module<Derived>::train() {
  impl_->train();
}

}  // namespace lmp::nets

#define LMP_DEFINE_MODULE_IMPL(module, impl) \
  struct module : public Module<impl> {      \
    using Module<impl>::Module;              \
  };
#define LMP_DEFINE_MODULE(module) LMP_DEFINE_MODULE_IMPL(module, module##Impl)