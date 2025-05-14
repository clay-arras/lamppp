#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include "include/lamppp/tensor/align_utils.hpp"
#include "include/lamppp/tensor/data_ptr.hpp"

#ifdef __cplusplus

namespace lmp::tensor::detail::cuda {

constexpr size_t nArgs = 2;

namespace internal {

struct ArgStrides {
  DataPtr stride1;
  DataPtr stride2;
};
struct OffsetPair {
  size_t offset1;
  size_t offset2;
};

}  // namespace internal

// TODO: in the future maybe do template <size_t nArgs>
class OffsetUtil {
 public:
  explicit OffsetUtil(const shape_list& a_shape, const shape_list& b_shape,
                      const stride_list& a_stride, const stride_list& b_stride,
                      const shape_list& out_shape);
  __device__ internal::OffsetPair get(size_t idx) const;

  internal::ArgStrides arg_strides_;
  DataPtr out_shape_;
  size_t ndim;

 private:
  std::vector<stride_t> init_padded_strides_(
      const std::vector<size_t>& shape, const std::vector<stride_t>& stride);
};

};  // namespace lmp::tensor::detail::cuda

#endif