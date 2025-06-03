#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda/std/array>
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cuda/offset_util.cuh"
#include "lamppp/tensor/cuda/ptr_pack.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

constexpr size_t NArgs = ExpandMetaHandler::NumElem;

/// @internal
template <typename PtrList, typename OpFn>
__global__ void vectorized_expand_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                         const CUDAOffsetUtil<NArgs>* align);

template <typename PtrList, typename OpFn>
void expand_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            const CUDAOffsetUtil<NArgs>* align);

template <template <typename> class OpFunctor, typename... Args>
void expand_dispatch_handler(ExpandMetaHandler& meta, Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg1_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(meta.in()[1]->type(), [&] {
        using arg2_dtype_t = scalar_t;
        expand_kernel_launcher(
            internal::CUDAPtrPack<out_dtype_t, arg1_dtype_t, arg2_dtype_t>(
                static_cast<out_dtype_t*>(meta.out().data()),
                static_cast<arg1_dtype_t*>(meta.in()[0]->data()),
                static_cast<arg2_dtype_t*>(meta.in()[1]->data())),
            OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
            meta.out().numel(),
            static_cast<const CUDAOffsetUtil<NArgs>*>(meta.offset()));
      });
    });
  });
}
/// @endinternal

}  // namespace lmp::tensor::detail::cuda