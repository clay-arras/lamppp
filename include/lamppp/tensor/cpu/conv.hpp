#pragma once

#include <cstddef>

namespace lmp::tensor::detail::cpu {

template <typename U, typename V, typename OutType>
void cpuConv2d(const U* input, const V* kernel, OutType* output, size_t stride, size_t padding,
                size_t dilation, const size_t* input_shape, const size_t* kernel_shape, const size_t* output_shape);

}  // namespace lmp::tensor::detail::cpu

