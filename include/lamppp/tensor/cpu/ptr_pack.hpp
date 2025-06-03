#pragma once

#include <array>
#include <tuple>

namespace lmp::tensor::detail::cpu::internal {

/// @internal
template <typename U, typename V>
struct TransformFunctor {
  U operator()(void* p, std::size_t i) const {
    return static_cast<U>(static_cast<V*>(p)[i]);
  }
};
/// @endinternal

/// @internal
/**
 * @brief Pack of pointers for CPU
 * @details its a list of void* and a corresponding functions to get the
 * value at indices i, in the output type.
 */
template <class OutT, class... SrcTs>
class PtrPack {
 public:
  static constexpr std::size_t N = sizeof...(SrcTs);

  ::std::array<void*, N + 1> data;
  ::std::tuple<TransformFunctor<OutT, OutT>, TransformFunctor<OutT, SrcTs>...>
      fns;

  constexpr PtrPack(OutT* out, SrcTs*... in)
      : data{static_cast<void*>(out), static_cast<void*>(in)...},
        fns{TransformFunctor<OutT, OutT>{},
            TransformFunctor<OutT, SrcTs>{}...} {}

  void set_Out(std::size_t idx, OutT value) {
    static_cast<OutT*>(data[0])[idx] = value;
  }
};
/// @endinternal

}  // namespace lmp::tensor::detail::cpu::internal