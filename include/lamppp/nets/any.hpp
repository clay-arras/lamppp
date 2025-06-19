#pragma once

#include <any>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include "lamppp/common/assert.hpp"

namespace lmp::nets {

class AnyModule {
 public:
  class Placeholder {
   public:
    virtual std::any call(const std::vector<std::any>& args) = 0;
    virtual ~Placeholder() = default;
  };

  template <typename Module, typename Ret, typename... Args>
  class Holder : public Placeholder {
    using FuncPtr = Ret (Module::*)(Args...);

   public:
    explicit Holder(std::shared_ptr<Module> mod, FuncPtr forward)
        : mod_(mod), forward_(forward) {};
    ~Holder() override = default;
    std::any call(const std::vector<std::any>& args) override {
      LMP_CHECK(args.size() == sizeof...(Args)) << "Invalid forward arguments";
      return invoke(args, std::index_sequence_for<Args...>{});
    }

   private:
    template <size_t... Idx>
    std::any invoke(const std::vector<std::any>& args,
                    std::index_sequence<Idx...> seq) {
      return std::any(static_cast<Module*>(mod_)->*forward_(
          any_cast<Args>(args[seq[Idx]])...));
    }

    std::shared_ptr<Module> mod_;
    FuncPtr forward_;
  };

  template <typename Module, typename Ret, typename... Args>
  explicit AnyModule(Module mod, Ret (Module::*forward)(Args...)) {
    impl_ = std::make_unique<Holder<Module, Ret, Args...>>(
        std::make_shared(mod), forward);
  }

  std::any call(const std::vector<std::any>& args) const {
    return impl_->call(args);
  }

 private:
  std::unique_ptr<Placeholder> impl_;
};

}