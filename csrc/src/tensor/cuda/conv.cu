#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/cuda/conv.cuh"
#include "lamppp/tensor/cuda/list_ptr.cuh"
#include "lamppp/tensor/data_type.hpp"

namespace lmp::tensor::detail::cuda {

using ssize_t = ptrdiff_t;  // signed size_t

namespace {
template <typename U, typename V, typename OutType>
__global__ void cudaConv1dKernel(const U* input, const V* kernel,
                                 OutType* output, size_t stride, size_t padding,
                                 size_t dilation, const size_t* input_shape,
                                 const size_t* kernel_shape,
                                 const size_t* output_shape) {
  size_t i = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (i >= output_shape[0]) {
    return;
  }

  OutType sum = 0;
  ssize_t start_i = (i * stride) - padding;
  for (size_t ii = 0; ii < kernel_shape[0]; ii++) {
    ssize_t idx_i = start_i + (dilation * ii);
    if (idx_i >= 0 && idx_i < input_shape[0]) {
      sum +=
          static_cast<OutType>(kernel[ii]) * static_cast<OutType>(input[idx_i]);
    }
  }
  output[i] = sum;
}

template <typename U, typename V, typename OutType>
__global__ void cudaConv2dKernel(const U* input, const V* kernel,
                                 OutType* output, size_t stride, size_t padding,
                                 size_t dilation, const size_t* input_shape,
                                 const size_t* kernel_shape,
                                 const size_t* output_shape) {
  size_t i = (blockDim.x * blockIdx.x) + threadIdx.x;
  size_t j = (blockDim.y * blockIdx.y) + threadIdx.y;

  if (i >= output_shape[0] || j >= output_shape[1]) {
    return;
  }

  OutType sum =
      0;  // TODO(nx2372): optimization, use copied values instead of arbitrary size arrays?
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
__global__ void cudaConv3dKernel(const U* input, const V* kernel,
                                 OutType* output, size_t stride, size_t padding,
                                 size_t dilation, const size_t* input_shape,
                                 const size_t* kernel_shape,
                                 const size_t* output_shape) {
  size_t i = (blockDim.x * blockIdx.x) + threadIdx.x;
  size_t j = (blockDim.y * blockIdx.y) + threadIdx.y;
  size_t k = (blockDim.z * blockIdx.z) + threadIdx.z;

  if (i >= output_shape[0] || j >= output_shape[1] || k >= output_shape[2]) {
    return;
  }

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
          sum += static_cast<OutType>(
                     kernel[(ii * kernel_shape[1] * kernel_shape[2]) +
                            (jj * kernel_shape[2]) + kk]) *
                 static_cast<OutType>(
                     input[(idx_i * input_shape[1] * input_shape[2]) +
                           (idx_j * input_shape[2]) + idx_k]);
        }
      }
    }
  }
  output[(i * output_shape[1] * output_shape[2]) + (j * output_shape[2]) + k] =
      sum;
}

}  // namespace

template <typename U, typename V, typename OutType>
void cudaConv1d(const U* input, const V* kernel, OutType* output, size_t stride,
                size_t padding, size_t dilation, const size_t* input_shape,
                const size_t* kernel_shape, const size_t* output_shape) {

  dim3 threads(256);
  dim3 blocks((output_shape[0] + threads.x - 1) / threads.x);
  ListDevicePtr<size_t> d_input_shape(input_shape, 1);
  ListDevicePtr<size_t> d_kernel_shape(kernel_shape, 1);
  ListDevicePtr<size_t> d_output_shape(output_shape, 1);
  cudaConv1dKernel<U, V, OutType><<<blocks, threads>>>(
      input, kernel, output, stride, padding, dilation, d_input_shape.get(),
      d_kernel_shape.get(), d_output_shape.get());
}

template <typename U, typename V, typename OutType>
void cudaConv2d(const U* input, const V* kernel, OutType* output, size_t stride,
                size_t padding, size_t dilation, const size_t* input_shape,
                const size_t* kernel_shape, const size_t* output_shape) {

  dim3 threads(16, 16);
  dim3 blocks((output_shape[0] + threads.x - 1) / threads.x,
              (output_shape[1] + threads.y - 1) / threads.y);
  ListDevicePtr<size_t> d_input_shape(input_shape, 2);
  ListDevicePtr<size_t> d_kernel_shape(kernel_shape, 2);
  ListDevicePtr<size_t> d_output_shape(output_shape, 2);
  cudaConv2dKernel<U, V, OutType><<<blocks, threads>>>(
      input, kernel, output, stride, padding, dilation, d_input_shape.get(),
      d_kernel_shape.get(),
      d_output_shape
          .get());  // TODO(nx2372): need to listPtr the shapes to device
}

template <typename U, typename V, typename OutType>
void cudaConv3d(const U* input, const V* kernel, OutType* output, size_t stride,
                size_t padding, size_t dilation, const size_t* input_shape,
                const size_t* kernel_shape, const size_t* output_shape) {

  dim3 threads(8, 8, 8);
  dim3 blocks((output_shape[0] + threads.x - 1) / threads.x,
              (output_shape[1] + threads.y - 1) / threads.y,
              (output_shape[2] + threads.z - 1) / threads.z);
  ListDevicePtr<size_t> d_input_shape(input_shape, 3);
  ListDevicePtr<size_t> d_kernel_shape(kernel_shape, 3);
  ListDevicePtr<size_t> d_output_shape(output_shape, 3);
  cudaConv3dKernel<U, V, OutType><<<blocks, threads>>>(
      input, kernel, output, stride, padding, dilation, d_input_shape.get(),
      d_kernel_shape.get(), d_output_shape.get());
}

#define INSTANTIATE_CONV1D(arg1_type, arg2_type, out_type)                   \
  template void cudaConv1d<arg1_type, arg2_type, out_type>(                  \
      const arg1_type*, const arg2_type*, out_type*, size_t, size_t, size_t, \
      const size_t*, const size_t*, const size_t*);

#define INSTANTIATE_CONV2D(arg1_type, arg2_type, out_type)                   \
  template void cudaConv2d<arg1_type, arg2_type, out_type>(                  \
      const arg1_type*, const arg2_type*, out_type*, size_t, size_t, size_t, \
      const size_t*, const size_t*, const size_t*);

#define INSTANTIATE_CONV3D(arg1_type, arg2_type, out_type)                   \
  template void cudaConv3d<arg1_type, arg2_type, out_type>(                  \
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

}  // namespace lmp::tensor::detail::cuda
