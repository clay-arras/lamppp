#pragma once

#include <vector>
#include "lamppp/tensor/data_type.hpp"

/// @todo: can be increased
enum { LMP_MAX_DIMS = 16 };

/// @internal
namespace lmp::tensor::detail {

using stride_t = int64_t;
using stride_list = std::vector<stride_t>;
using shape_list = std::vector<size_t>;

/**
 * @brief Utility class for aligning shapes of tensors
 * 
 * @details This class is used to align the shapes of two tensors so that they can be broadcasted together.
 * Alignment is done by NumPy's broadcasting rules.
 * @note This class is used internally by the tensor library and is not intended to be used by the user.
 * 
 */
class AlignUtil {
 public:
  explicit AlignUtil(const std::vector<size_t>& a_shape,
                     const std::vector<size_t>& b_shape);

  std::vector<size_t> aligned_shape_;
  std::vector<stride_t> aligned_stride_;
  size_t aligned_size_;

 private:
  static std::vector<size_t> calc_aligned_shape(
      const std::vector<size_t>& a_shape, const std::vector<size_t>& b_shape);
  std::vector<stride_t> calc_aligned_stride();
};
/// @endinternal

}  // namespace lmp::tensor::detail