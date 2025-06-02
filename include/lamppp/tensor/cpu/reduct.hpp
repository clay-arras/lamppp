#pragma once

#include "lamppp/tensor/cpu/kernels.hpp"
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cpu/ptr_pack.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

template <typename PtrList, typename OpFn>
void vectorized_reduct_kernel(PtrList ptr_, OpFn fn_, size_t i,
                                         size_t axis, const size_t* shape,
                                         const stride_t* strides);

template <typename PtrList, typename OpFn>
void reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size, size_t axis,
                            const size_t* shape, const stride_t* strides,
                            const size_t ndims);

template <template <typename> class OpFunctor, typename... Args>
void reduct_dispatch_handler(ReductMetaHandler& meta, size_t axis,
                             Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg_dtype_t = scalar_t;
      reduct_kernel_launcher(
          internal::PtrPack<out_dtype_t, arg_dtype_t>(
              static_cast<out_dtype_t*>(meta.out().data()),
              static_cast<arg_dtype_t*>(meta.in()[0]->data())),
          OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
          meta.out().numel(), axis, meta.in()[0]->shape().data(),
          meta.in()[0]->strides().data(), meta.out().shape().size());
    });
  });
}

}  // namespace lmp::tensor::detail::cpu