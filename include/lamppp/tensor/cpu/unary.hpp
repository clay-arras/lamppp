#pragma once

#include <cstddef>
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cpu/ptr_pack.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

/// @internal
template <typename PtrList, typename OpFn>
void vectorized_unary_kernel(PtrList ptr_, OpFn fn_, size_t i);

template <typename PtrList, typename OpFn>
void unary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size);

template <template <typename> class OpFunctor, typename... Args>
void unary_dispatch_handler(UnaryMetaHandler& meta, Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg_dtype_t = scalar_t;
      unary_kernel_launcher(
          internal::PtrPack<out_dtype_t, arg_dtype_t>(
              static_cast<out_dtype_t*>(meta.out().data()),
              static_cast<arg_dtype_t*>(meta.in()[0]->data())),
          OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
          meta.out().numel());
    });
  });
}
/// @endinternal

}  // namespace lmp::tensor::detail::cpu