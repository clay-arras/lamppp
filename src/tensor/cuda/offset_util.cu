#include <cassert>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/cuda/offset_util.cuh"

namespace lmp::tensor::detail::cuda {

OffsetUtil::OffsetUtil(const shape_list& a_shape, const shape_list& b_shape,
                       const stride_list& a_stride, const stride_list& b_stride,
                       const stride_list& out_stride, size_t ndim)
    : ndim(ndim) {

  std::vector<stride_t> a_stride_tmp = init_padded_strides_(a_shape, a_stride);
  std::vector<stride_t> b_stride_tmp = init_padded_strides_(b_shape, b_stride);

  arg_strides_[0] =
      ListDevicePtr<stride_t>(a_stride_tmp.data(), a_stride_tmp.size());
  arg_strides_[1] =
      ListDevicePtr<stride_t>(b_stride_tmp.data(), b_stride_tmp.size());
  arg_strides_[2] =
      ListDevicePtr<stride_t>(out_stride.data(), out_stride.size());

  arg_pointers_[0] = arg_strides_[0].get();
  arg_pointers_[1] = arg_strides_[1].get();
  arg_pointers_[2] = arg_strides_[2].get();
}

__device__ ::cuda::std::array<stride_t, NVARS> OffsetUtil::get(
    size_t idx) const {
  ::cuda::std::array<stride_t, NVARS> result;
  result[0] = 0;
  result[1] = 0;
  result[2] = idx;

  for (size_t i = 0; i < ndim; ++i) {
    stride_t this_idx = idx / static_cast<const stride_t*>(arg_pointers_[2])[i];
    idx = idx % static_cast<const stride_t*>(arg_pointers_[2])[i];

    result[0] += (this_idx * static_cast<const stride_t*>(arg_pointers_[0])[i]);
    result[1] += (this_idx * static_cast<const stride_t*>(arg_pointers_[1])[i]);
  }

  return result;
}

std::vector<stride_t> OffsetUtil::init_padded_strides_(
    const std::vector<size_t>& shape, const std::vector<stride_t>& stride) {
  assert(ndim > 0);
  assert(shape.size() <= ndim);
  assert(shape.size() == stride.size());

  std::vector<stride_t> padded(ndim, 0);
  const size_t from_back = shape.size();

  for (size_t k = 0; k < from_back; ++k) {
    size_t dst = ndim - 1 - k;
    size_t src = from_back - 1 - k;

    if (shape[src] != 1) {
      padded[dst] = stride[src];
    }
  }
  return padded;
}

}  // namespace lmp::tensor::detail::cuda