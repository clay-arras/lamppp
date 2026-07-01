#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cuda/std/array>
#include "lamp3/tensor/utils/align_utils.hpp"
#include "lamp3/tensor/cpu/offset_util.hpp"
#include "lamp3/tensor/cuda/list_ptr.cuh"
#include "lamp3/tensor/tensor_impl.hpp"

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

/// @internal
/**
 * @brief Offset utility for CUDA
 * @details see OffsetUtil.hpp for more details
 */
template <size_t NArgs>
class CUDAOffsetUtil : public OffsetUtil {
 public:
  explicit CUDAOffsetUtil(::std::array<const TensorImpl*, NArgs> ins,
                          const TensorImpl& outs);
  __device__ ::cuda::std::array<stride_t, NArgs + 1> get(size_t idx) const;

  ::cuda::std::array<ListDevicePtr<stride_t>, NArgs + 1> arg_strides_;
  ::cuda::std::array<void*, NArgs + 1> arg_pointers_;
};

namespace {
template <size_t NArgs>
std::unique_ptr<OffsetUtil> offset_util_cuda(::std::array<const TensorImpl*, NArgs> ins, 
    const TensorImpl& out) {
  return std::make_unique<CUDAOffsetUtil<NArgs>>(ins, out);
}
}
/// @endinternal

};  // namespace lmp::tensor::detail::cuda

#endif