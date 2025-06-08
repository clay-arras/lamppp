#pragma once

#include "lamppp/tensor/cpu/kernels.hpp"
#include "lamppp/tensor/cpu/binary.hpp"
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cpu/offset_util.hpp"
#include "lamppp/tensor/cpu/ptr_pack.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

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
void expand_dispatch_handler(BinaryMetaHandler& meta, Args&&... args);
/// @endinternal

extern template void expand_dispatch_handler<AddFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<SubFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<MulFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<DivFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<PowFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<EqFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<NeFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<GeFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<GtFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<LeFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<LtFunctor>(BinaryMetaHandler&);

}  // namespace lmp::tensor::detail::cpu