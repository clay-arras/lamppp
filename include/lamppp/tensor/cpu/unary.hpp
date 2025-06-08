#pragma once

#include <cstddef>
#include "lamppp/tensor/cpu/kernels.hpp"
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
void unary_dispatch_handler(UnaryMetaHandler& meta, Args&&... args); 

extern template void unary_dispatch_handler<NegFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<ExpFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<LogFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<SqrtFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<AbsFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<SinFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<CosFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<TanFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<ClampFunctor>(UnaryMetaHandler&, Scalar&,
                                                   Scalar&);

/// @endinternal

}  // namespace lmp::tensor::detail::cpu