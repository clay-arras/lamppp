#include <cassert>
#include "include/lamppp/tensor/align_utils.hpp"
#include "include/lamppp/tensor/cuda/offset_util.cuh"
#include "include/lamppp/tensor/data_ptr.hpp"
#include "include/lamppp/tensor/data_type.hpp"
#include "include/lamppp/tensor/device_type.hpp"
#include "include/lamppp/tensor/native/copy.cuh"
#include "include/lamppp/tensor/native/empty.cuh"

namespace lmp::tensor::detail::cuda {

OffsetUtil::OffsetUtil(const shape_list& a_shape, const shape_list& b_shape,
                       const stride_list& a_stride, const stride_list& b_stride,
                       const shape_list& out_shape)
    : ndim(out_shape.size()) {

  std::vector<stride_t> a_stride_tmp = init_padded_strides_(a_shape, a_stride);
  std::vector<stride_t> b_stride_tmp = init_padded_strides_(b_shape, b_stride);

  arg_strides_.stride1 = native::empty_stub(
      DeviceType::CUDA, a_stride_tmp.size() * sizeof(stride_t));
  arg_strides_.stride2 = native::empty_stub(
      DeviceType::CUDA, b_stride_tmp.size() * sizeof(stride_t));
  native::copy_stub(DeviceType::CPU, DeviceType::CUDA, a_stride_tmp.data(),
                    arg_strides_.stride1.data, a_stride_tmp.size(),
                    TypeMeta<stride_t>::value, TypeMeta<stride_t>::value);
  native::copy_stub(DeviceType::CPU, DeviceType::CUDA, b_stride_tmp.data(),
                    arg_strides_.stride2.data, b_stride_tmp.size(),
                    TypeMeta<stride_t>::value, TypeMeta<stride_t>::value);
  out_shape_ =
      native::empty_stub(DeviceType::CUDA, out_shape.size() * sizeof(size_t));
  cudaMemcpy(out_shape_.data, out_shape.data(),
             out_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);
}

__device__ internal::OffsetPair OffsetUtil::get(size_t idx) const {
  internal::OffsetPair result;
  result.offset1 = 0;
  result.offset2 = 0;

  for (size_t i = 0; i < ndim; ++i) {
    stride_t this_idx = idx % static_cast<const size_t*>(out_shape_.data)[i];
    idx = idx / static_cast<const size_t*>(out_shape_.data)[i];

    result.offset1 +=
        this_idx * static_cast<const stride_t*>(arg_strides_.stride1.data)[i];
    result.offset2 +=
        this_idx * static_cast<const stride_t*>(arg_strides_.stride2.data)[i];
  }

  return result;
}

std::vector<stride_t> OffsetUtil::init_padded_strides_(
    const std::vector<size_t>& shape, const std::vector<stride_t>& stride) {
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