#include "lamppp/tensor/align_utils.hpp"
#include <iostream>
#include <numeric>
#include "lamppp/common/assert.hpp"

namespace lmp::tensor::detail {

AlignUtil::AlignUtil(const shape_list& a_shape, const shape_list& b_shape)
    : aligned_shape_(calc_aligned_shape(a_shape, b_shape)),
      aligned_stride_(calc_aligned_stride()),
      aligned_size_(aligned_shape_.empty()
                        ? 0
                        : std::accumulate(aligned_shape_.begin(),
                                          aligned_shape_.end(), 1,
                                          std::multiplies<>())) {}

std::vector<size_t> AlignUtil::calc_aligned_shape(
    const std::vector<size_t>& a_shape, const std::vector<size_t>& b_shape) {
  size_t out_dims = std::max(a_shape.size(), b_shape.size());
  std::vector<size_t> out_shape(out_dims);
  LMP_CHECK(out_dims <= LMP_MAX_DIMS) << "Too many dims";

#pragma unroll
  for (size_t i = LMP_MAX_DIMS; i-- > 0;) {
    if (i >= out_dims)
      continue;

    int offset = out_dims - 1 - i;  // needs to use signed int for this part
    int a_idx = a_shape.size() - 1 - offset;
    int b_idx = b_shape.size() - 1 - offset;

    int a_val = (a_idx >= 0 ? a_shape[a_idx] : 1);
    int b_val = (b_idx >= 0 ? b_shape[b_idx] : 1);

    LMP_CHECK(a_val == 1 || b_val == 1 || a_val == b_val) <<
              "calc_aligned_shape: Broadcast rule violation.";
    out_shape[i] = (a_val != 1 ? a_val : b_val);
  }
  return out_shape;
}

std::vector<stride_t> AlignUtil::calc_aligned_stride() {
  stride_t stride = 1;
  std::vector<stride_t> strides(aligned_shape_.size());
  for (int i = aligned_shape_.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= aligned_shape_[i];
  }
  return strides;
}

}  // namespace lmp::tensor::detail