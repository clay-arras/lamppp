#pragma once

#include "lamppp/tensor/cpu/kernels.hpp"
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cpu/ptr_pack.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

/// @internal
template <typename PtrList, typename OpFn>
void vectorized_reduct_kernel(PtrList ptr_, OpFn fn_, size_t i,
                                         size_t axis, const size_t* shape,
                                         const stride_t* strides);

template <typename PtrList, typename OpFn>
void reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size, size_t axis,
                            const size_t* shape, const stride_t* strides,
                            size_t ndims);

template <template <typename> class OpFunctor, typename... Args>
void reduct_dispatch_handler(ReductMetaHandler& meta, size_t axis,
                             Args&&... args); 

extern template void reduct_dispatch_handler<SumFunctor>(ReductMetaHandler&, size_t);
extern template void reduct_dispatch_handler<MaxFunctor>(ReductMetaHandler&, size_t);
extern template void reduct_dispatch_handler<MinFunctor>(ReductMetaHandler&, size_t);
extern template void reduct_dispatch_handler<ProdFunctor>(ReductMetaHandler&, size_t);

/// @endinternal

}  // namespace lmp::tensor::detail::cpu