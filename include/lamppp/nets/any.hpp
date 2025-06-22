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
  template <typename Derived>
  explicit AnyModule(Module<Derived> mod) {
    impl_ = make_holder(detail::UnsafeModuleAccessor<Derived>::getImpl(mod),
                        &Derived::forward);
  }

  std::shared_ptr<ModuleImpl> getImpl();
  std::any call(const std::vector<std::any>& args) const; 

 private:
  class Placeholder;
  template <typename MImpl, typename... Args>
  class Holder;

  std::shared_ptr<Placeholder> impl_;

 protected:
  template <typename Impl, typename R, typename... Args>
  std::shared_ptr<AnyModule::Placeholder> make_holder(std::shared_ptr<Impl> m,
                                                      R (Impl::*fp)(Args...)
                                                          const) {
    using H = typename AnyModule::Holder<Impl, Args...>;
    return std::make_shared<H>(std::move(m), fp);
  }
};


class AnyModule::Placeholder {
  public:
  virtual ~Placeholder() = default;

  virtual std::any call(const std::vector<std::any>& args) = 0;
  virtual std::shared_ptr<ModuleImpl> getImpl() = 0;
};


template <typename MImpl, typename... Args>
class AnyModule::Holder : public AnyModule::Placeholder {
  using FuncPtr = std::invoke_result_t<decltype(&MImpl::forward), MImpl*,
                                        Args...> (MImpl::*)(Args...) const;

  public:
  ~Holder() override = default;
  explicit Holder(std::shared_ptr<MImpl> mod, FuncPtr forward)
      : mod_(mod), forward_(forward) {};

  std::any call(const std::vector<std::any>& args) override {
    LMP_CHECK(args.size() == sizeof...(Args)) << "Invalid forward arguments";
    return invoke(args, std::index_sequence_for<Args...>{});
  }
  std::shared_ptr<ModuleImpl> getImpl() override { return mod_; };

  private:
  template <size_t... Idx>
  std::any invoke(const std::vector<std::any>& args,
                  std::index_sequence<Idx...> /*seq*/) {
    return std::any((static_cast<MImpl*>(mod_.get())->*forward_)(
        any_cast<Args>(args[Idx])...));
  }

  std::shared_ptr<MImpl> mod_;
  FuncPtr forward_;
};

}  // namespace lmp::nets