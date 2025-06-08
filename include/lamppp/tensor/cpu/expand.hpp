#pragma once

#include "lamppp/tensor/cpu/kernels.hpp"
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cpu/offset_util.hpp"
#include "lamppp/tensor/cpu/ptr_pack.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

constexpr size_t kNArgs = ExpandMetaHandler::kNumElem;

/// @internal
template <typename PtrList, typename OpFn>
void vectorized_expand_kernel(PtrList ptr_, OpFn fn_, size_t i,
                              const CPUOffsetUtil<kNArgs>* align);

template <typename PtrList, typename OpFn>
void expand_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            const CPUOffsetUtil<kNArgs>* align);
/// @endinternal

/// @internal
template <template <typename> class OpFunctor, typename... Args>
void expand_dispatch_handler(ExpandMetaHandler& meta, Args&&... args);
/// @endinternal

extern template void expand_dispatch_handler<AddFunctor>(ExpandMetaHandler&);
extern template void expand_dispatch_handler<SubFunctor>(ExpandMetaHandler&);
extern template void expand_dispatch_handler<MulFunctor>(ExpandMetaHandler&);
extern template void expand_dispatch_handler<DivFunctor>(ExpandMetaHandler&);
extern template void expand_dispatch_handler<PowFunctor>(ExpandMetaHandler&);
extern template void expand_dispatch_handler<EqFunctor>(ExpandMetaHandler&);
extern template void expand_dispatch_handler<NeFunctor>(ExpandMetaHandler&);
extern template void expand_dispatch_handler<GeFunctor>(ExpandMetaHandler&);
extern template void expand_dispatch_handler<GtFunctor>(ExpandMetaHandler&);
extern template void expand_dispatch_handler<LeFunctor>(ExpandMetaHandler&);
extern template void expand_dispatch_handler<LtFunctor>(ExpandMetaHandler&);

}  // namespace lmp::tensor::detail::cpu