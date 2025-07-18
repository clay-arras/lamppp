#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/cpu/conv.hpp"
#include "lamppp/tensor/data_type.hpp"

namespace lmp::tensor::detail::cpu {

using ssize_t = ptrdiff_t; // signed size_t

namespace {
template <typename U, typename V, typename OutType>
void cpuConv1dKernel(const U* input, const V* kernel, OutType* output, size_t stride, size_t padding,
                size_t dilation, const size_t* input_shape, const size_t* kernel_shape, const size_t* output_shape, size_t i) {
    OutType sum = 0;
    ssize_t start_i = (i * stride) - padding;
    for (size_t ii = 0; ii < kernel_shape[0]; ii++) {
      ssize_t idx_i = start_i + (dilation * ii);
      if (idx_i >= 0 && idx_i < input_shape[0]) {
        sum += static_cast<OutType>(kernel[ii]) *
               static_cast<OutType>(input[idx_i]);
      }
    }
    output[i] = sum;
}

template <typename U, typename V, typename OutType>
void cpuConv2dKernel(const U* input, const V* kernel, OutType* output, size_t stride, size_t padding,
                size_t dilation, const size_t* input_shape, const size_t* kernel_shape, const size_t* output_shape, size_t i, size_t j) {
    OutType sum = 0; // TODO(nx2372): optimization, use shared memory instead of global reads
    ssize_t start_i = (i * stride) - padding;
    ssize_t start_j = (j * stride) - padding;
    for (size_t ii = 0; ii < kernel_shape[0]; ii++) {
        for (size_t jj = 0; jj < kernel_shape[1]; jj++) {
          ssize_t idx_i = start_i + (dilation * ii);
          ssize_t idx_j = start_j + (dilation * jj);
          if (idx_i >= 0 && idx_i < input_shape[0] && idx_j >= 0 &&
              idx_j < input_shape[1]) {
            sum += static_cast<OutType>(kernel[(ii * kernel_shape[1]) + jj]) *
                   static_cast<OutType>(input[(idx_i * input_shape[1]) + idx_j]);
          }
        }
    }
    output[(i * output_shape[1]) + j] = sum;
}

template <typename U, typename V, typename OutType>
void cpuConv3dKernel(const U* input, const V* kernel, OutType* output, size_t stride, size_t padding,
                size_t dilation, const size_t* input_shape, const size_t* kernel_shape, const size_t* output_shape, size_t i, size_t j, size_t k) {
    OutType sum = 0;
    ssize_t start_i = (i * stride) - padding;
    ssize_t start_j = (j * stride) - padding;
    ssize_t start_k = (k * stride) - padding;
    for (size_t ii = 0; ii < kernel_shape[0]; ii++) {
        for (size_t jj = 0; jj < kernel_shape[1]; jj++) {
            for (size_t kk = 0; kk < kernel_shape[2]; kk++) {
                ssize_t idx_i = start_i + (dilation * ii);
                ssize_t idx_j = start_j + (dilation * jj);
                ssize_t idx_k = start_k + (dilation * kk);
                if (idx_i >= 0 && idx_i < input_shape[0] && idx_j >= 0 &&
                    idx_j < input_shape[1] && idx_k >= 0 && idx_k < input_shape[2]) {
                    sum += static_cast<OutType>(kernel[(ii * kernel_shape[1] * kernel_shape[2]) + (jj * kernel_shape[2]) + kk]) *
                           static_cast<OutType>(input[(idx_i * input_shape[1] * input_shape[2]) + (idx_j * input_shape[2]) + idx_k]);
                }
            }
        }
    }
    output[(i * output_shape[1] * output_shape[2]) + (j * output_shape[2]) + k] = sum;
}

}  // namespace

template <typename U, typename V, typename OutType>
void cpuConv1d(const U* input, const V* kernel, OutType* output, size_t stride, size_t padding,
                size_t dilation, const size_t* input_shape, const size_t* kernel_shape, const size_t* output_shape) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < output_shape[0]; i++)
        cpuConv1dKernel<U, V, OutType>(input, kernel, output, stride, padding, dilation,
                                    input_shape, kernel_shape, output_shape, i);
}

template <typename U, typename V, typename OutType>
void cpuConv2d(const U* input, const V* kernel, OutType* output, size_t stride, size_t padding,
                size_t dilation, const size_t* input_shape, const size_t* kernel_shape, const size_t* output_shape) {
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < output_shape[0]; i++)
    for (size_t j = 0; j < output_shape[1]; j++)
        cpuConv2dKernel<U, V, OutType>(input, kernel, output, stride, padding, dilation,
                                    input_shape, kernel_shape, output_shape, i, j);
}

template <typename U, typename V, typename OutType>
void cpuConv3d(const U* input, const V* kernel, OutType* output, size_t stride, size_t padding,
                size_t dilation, const size_t* input_shape, const size_t* kernel_shape, const size_t* output_shape) {
#pragma omp parallel for collapse(3) schedule(static)
  for (size_t i = 0; i < output_shape[0]; i++)
    for (size_t j = 0; j < output_shape[1]; j++)
        for (size_t k = 0; k < output_shape[2]; k++)
            cpuConv3dKernel<U, V, OutType>(input, kernel, output, stride, padding, dilation,
                                        input_shape, kernel_shape, output_shape, i, j, k);
}

#define INSTANTIATE_CONV1D(arg1_type, arg2_type, out_type)                   \
  template void cpuConv1d<arg1_type, arg2_type, out_type>(                   \
      const arg1_type*, const arg2_type*, out_type*, size_t, size_t, size_t, \
      const size_t*, const size_t*, const size_t*);

#define INSTANTIATE_CONV2D(arg1_type, arg2_type, out_type)                   \
  template void cpuConv2d<arg1_type, arg2_type, out_type>(                   \
      const arg1_type*, const arg2_type*, out_type*, size_t, size_t, size_t, \
      const size_t*, const size_t*, const size_t*);

#define INSTANTIATE_CONV3D(arg1_type, arg2_type, out_type)                   \
  template void cpuConv3d<arg1_type, arg2_type, out_type>(                   \
      const arg1_type*, const arg2_type*, out_type*, size_t, size_t, size_t, \
      const size_t*, const size_t*, const size_t*);

LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_CONV1D, LMP_LIST_TYPES,
                               LMP_LIST_TYPES, LMP_LIST_TYPES);
LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_CONV2D, LMP_LIST_TYPES,
                               LMP_LIST_TYPES, LMP_LIST_TYPES);
LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_CONV3D, LMP_LIST_TYPES,
                               LMP_LIST_TYPES, LMP_LIST_TYPES);

#undef INSTANTIATE_CONV1D
#undef INSTANTIATE_CONV2D
#undef INSTANTIATE_CONV3D

}  // namespace lmp::tensor::detail::cpu
