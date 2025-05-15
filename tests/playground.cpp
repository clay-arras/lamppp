#include <vector>
#include "include/lamppp/lamppp.hpp"
#include "include/lamppp/tensor/align_utils.hpp"
#include "include/lamppp/tensor/cuda/offset_util.cuh"

int main() {
  auto a =
      lmp::tensor::Tensor(std::vector<int>{1, 2, 3}, std::vector<size_t>{3, 1});
  auto b = lmp::tensor::Tensor(std::vector<int>{4, 5}, std::vector<size_t>{2});

  auto c = a + b;
  std::cout << c << std::endl;
  // std::vector<size_t> a_shape{3, 1}, b_shape{2};
  // std::vector<int64_t> a_stride{1, 1}, b_stride{1};

  // std::vector<size_t> aligned_shape{3, 2};
  // std::vector<int64_t> aligned_stride{2, 1};

  // lmp::tensor::detail::cuda::OffsetUtil offset(
  //     a_shape, b_shape, a_stride, b_stride, aligned_shape, aligned_stride);
}