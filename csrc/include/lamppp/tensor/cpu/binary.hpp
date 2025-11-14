#pragma once

#include "lamppp/tensor/cpu/kernels.hpp"
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cpu/offset_util.hpp"
#include "lamppp/tensor/cpu/ptr_pack.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

constexpr size_t kNArgs = BinaryMetaHandler::kNumElem;

/// @internal
template <typename PtrList, typename OpFn>
void vectorized_binary_kernel(PtrList ptr_, OpFn fn_, size_t i);

template <typename PtrList, typename OpFn>
void binary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size);
/// @endinternal

/// @internal
template <template <typename> class OpFunctor, typename... Args>
void binary_dispatch_handler(BinaryMetaHandler& meta, Args&&... args);
/// @endinternal

extern template void binary_dispatch_handler<AddFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<SubFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<MulFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<DivFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<PowFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<EqFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<NeFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<GeFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<GtFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<LeFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<LtFunctor>(BinaryMetaHandler&);

}  // namespace lmp::tensor::detail::cpu