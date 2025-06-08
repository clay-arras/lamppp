#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda/std/array>
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cuda/kernels.cuh"
#include "lamppp/tensor/cuda/offset_util.cuh"
#include "lamppp/tensor/cuda/ptr_pack.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

constexpr size_t kNArgs = BinaryMetaHandler::kNumElem;

/// @internal
template <typename PtrList, typename OpFn>
__global__ void vectorized_binary_kernel(PtrList ptr_, OpFn fn_, size_t size);

template <typename PtrList, typename OpFn>
void binary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size);

template <template <typename> class OpFunctor, typename... Args>
void binary_dispatch_handler(BinaryMetaHandler& meta, Args&&... args);

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

/// @endinternal

}  // namespace lmp::tensor::detail::cuda