#pragma once

#include <array>
#include <type_traits>
#include "device_type.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/data_type.hpp"

namespace lmp::tensor::detail {

template <typename Fn>
struct DispatchStub {
  using fn_type = Fn;
  std::array<fn_type, static_cast<size_t>(DeviceType::Count)> table_{};
  constexpr DispatchStub() noexcept = default;

  void register_kernel(DeviceType dev, fn_type f) {
    table_[static_cast<size_t>(dev)] = f;
  }

  template <typename... Args>
  decltype(auto) operator()(DeviceType dev, Args&&... args) const {
    fn_type f = table_[static_cast<size_t>(dev)];
    LMP_INTERNAL_ASSERT(f) << "Kernel for this backend not registered";
    return f(std::forward<Args>(args)...);
  }
};

}  // namespace lmp::tensor::detail

#define LMP_DECLARE_DISPATCH(fn_type, stub_name) \
  ::lmp::tensor::detail::DispatchStub<fn_type>& stub_name();

#define LMP_DEFINE_DISPATCH(fn_type, stub_name)                      \
  ::lmp::tensor::detail::DispatchStub<fn_type>& stub_name() {        \
    static ::lmp::tensor::detail::DispatchStub<fn_type> s;           \
    return s;                                                        \
  };

#define LMP_REGISTER_DISPATCH(stub_name, dev, kernel_fn) \
  namespace {                                            \
  struct _Reg##kernel_fn {                               \
    _Reg##kernel_fn() {                                  \
      stub_name().register_kernel(dev, kernel_fn);       \
    }                                                    \
  } _auto_reg_##kernel_fn;                               \
  }
