#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cuda/std/array>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/cuda/list_ptr.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

using tensor_list = std::vector<lmp::tensor::TensorImpl>;

template <size_t NArgs>
class OffsetUtil {
 public:
  static constexpr size_t NVars = NArgs + 1;
  explicit OffsetUtil(::std::array<const TensorImpl*, NArgs> ins,
                      const TensorImpl& outs);
  __device__ ::cuda::std::array<stride_t, NArgs + 1> get(size_t idx) const;

  ::cuda::std::array<ListDevicePtr<stride_t>, NArgs + 1> arg_strides_;
  ::cuda::std::array<void*, NArgs + 1> arg_pointers_;
  size_t ndim;

 private:
  std::vector<stride_t> init_padded_strides_(
      const std::vector<size_t>& shape, const std::vector<stride_t>& stride);
};

};  // namespace lmp::tensor::detail::cuda

#endif