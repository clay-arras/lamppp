#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cuda/std/array>
#include <vector>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/cuda/list_ptr.cuh"

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

constexpr size_t NVARS = 3;

// TODO: in the future maybe do template <size_t nArgs>
class OffsetUtil {
 public:
  explicit OffsetUtil(const shape_list& a_shape, const shape_list& b_shape,
                      const stride_list& a_stride, const stride_list& b_stride,
                      const stride_list& out_stride, size_t ndims);
  __device__ ::cuda::std::array<stride_t, NVARS> get(size_t idx) const;

  ::cuda::std::array<ListDevicePtr<stride_t>, NVARS> arg_strides_;
  ::cuda::std::array<void*, NVARS> arg_pointers_;
  size_t ndim;

 private:
  std::vector<stride_t> init_padded_strides_(
      const std::vector<size_t>& shape, const std::vector<stride_t>& stride);
};

};  // namespace lmp::tensor::detail::cuda

#endif