#pragma once

#include <cassert>
#include <vector>
#include "include/lamppp/tensor/data_type.hpp"

#define LMP_MAX_DIMS 16  // TODO: can be increased

namespace lmp::tensor::detail {

using stride_t = int64_t;
using stride_list = std::vector<stride_t>;
using shape_list = std::vector<size_t>;

class AlignUtil {
 public:
  explicit AlignUtil(const std::vector<size_t>& a_shape,
                     const std::vector<size_t>& b_shape);

  std::vector<size_t> aligned_shape_;
  std::vector<stride_t> aligned_stride_;
  size_t aligned_size_;

 private:
  std::vector<size_t> calc_aligned_shape(const std::vector<size_t>& a_shape,
                                         const std::vector<size_t>& b_shape);
  std::vector<stride_t> calc_aligned_stride();
};

}  // namespace lmp::tensor::detail