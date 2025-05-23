#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cuda/std/array>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/cpu/offset_util.hpp"
#include "lamppp/tensor/cuda/list_ptr.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

template <size_t NArgs>
class CUDAOffsetUtil : public OffsetUtil<NArgs> {
 public:
  explicit CUDAOffsetUtil(::std::array<const TensorImpl*, NArgs> ins,
                          const TensorImpl& outs);
  __device__ ::cuda::std::array<stride_t, NArgs + 1> get(size_t idx) const;

  ::cuda::std::array<ListDevicePtr<stride_t>, NArgs + 1> arg_strides_;
  ::cuda::std::array<void*, NArgs + 1> arg_pointers_;
};

};  // namespace lmp::tensor::detail::cuda

#endif