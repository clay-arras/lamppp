#pragma once

#include <any>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include "lamppp/common/assert.hpp"
#include "lamppp/nets/module.hpp"

namespace lmp::nets {

class AnyModule {
 public:
  class Placeholder {
   public:
    virtual std::any call(const std::vector<std::any>& args) = 0;
    virtual ~Placeholder() = default;
  };

  template <typename ModuleImpl, typename... Args>
  class Holder : public Placeholder {
    using FuncPtr = std::invoke_result_t<decltype(&ModuleImpl::forward), ModuleImpl*,
                                         Args...> (ModuleImpl::*)(Args...) const;

   public:
    explicit Holder(std::shared_ptr<ModuleImpl> mod, FuncPtr forward)
        : mod_(mod), forward_(forward) {};
    ~Holder() override = default;
    std::any call(const std::vector<std::any>& args) override {
      LMP_CHECK(args.size() == sizeof...(Args)) << "Invalid forward arguments";
      return invoke(args, std::index_sequence_for<Args...>{});
    }

   private:
    template <size_t... Idx>
    std::any invoke(const std::vector<std::any>& args,
                    std::index_sequence<Idx...>  /*seq*/) {
      return std::any((static_cast<ModuleImpl*>(mod_.get())->*forward_)(
          any_cast<Args>(args[Idx])...));
    }

    std::shared_ptr<ModuleImpl> mod_;
    FuncPtr forward_;
  };

  template <typename Impl, typename R, typename... Args>
  std::shared_ptr<AnyModule::Placeholder> make_holder(std::shared_ptr<Impl> m,
                                                      R (Impl::*fp)(Args...) const) {
    using H = typename AnyModule::Holder<Impl, Args...>;
    return std::make_shared<H>(std::move(m), fp);
  }

  template <typename Derived>
  explicit AnyModule(Module<Derived> mod) {
    impl_ = make_holder(detail::UnsafeModuleAccessor<Derived>::getImpl(mod),
                        &Derived::forward);
  }

  std::any call(const std::vector<std::any>& args) const {
    return impl_->call(args);
  }

 private:
  std::shared_ptr<Placeholder> impl_;
};

}