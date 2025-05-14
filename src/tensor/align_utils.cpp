#include "include/lamppp/tensor/align_utils.hpp"
#include <cassert>
#include <numeric>

namespace lmp::tensor::detail {

AlignUtil::AlignUtil(const shape_list& a_shape, const shape_list& b_shape)
    : aligned_shape_(calc_aligned_shape(a_shape, b_shape)),
      aligned_size_(aligned_shape_.empty()
                        ? 0
                        : std::accumulate(aligned_shape_.begin(),
                                          aligned_shape_.end(), 1,
                                          std::multiplies<>())) {}

std::vector<size_t> AlignUtil::calc_aligned_shape(
    const std::vector<size_t>& a_shape, const std::vector<size_t>& b_shape) {
  size_t out_dims = std::max(a_shape.size(), b_shape.size());
  std::vector<size_t> out_shape(out_dims);
  assert(out_dims <= LMP_MAX_DIMS && "Too many dims");

#pragma unroll
  for (size_t i = LMP_MAX_DIMS; i-- > 0;) {
    if (i >= out_dims)
      continue;

    size_t offset = out_dims - 1 - i;
    size_t a_idx = a_shape.size() > offset ? a_shape.size() - 1 - offset : -1;
    size_t b_idx = b_shape.size() > offset ? b_shape.size() - 1 - offset : -1;

    size_t a_val = (a_idx >= 0 ? a_shape[a_idx] : 1);
    size_t b_val = (b_idx >= 0 ? b_shape[b_idx] : 1);

    assert(a_val == 1 || b_val == 1 || a_val == b_val);
    out_shape[i] = (a_val != 1 ? a_val : b_val);
  }
  return out_shape;
}

}  // namespace lmp::tensor::detail