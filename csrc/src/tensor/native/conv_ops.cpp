#include "lamppp/tensor/native/conv_ops.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(conv1d_fn, conv1d_stub);
LMP_DEFINE_DISPATCH(conv2d_fn, conv2d_stub);
LMP_DEFINE_DISPATCH(conv3d_fn, conv3d_stub);

Tensor conv1d(const Tensor& input, const Tensor& kernel, size_t stride,
              size_t padding, size_t dilation) {
  LMP_CHECK(input.device() == kernel.device())
      << "Tensors are on different devices";
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(conv1d_stub()(
          input.device(), *detail::UnsafeTensorAccessor::getImpl(input),
          *detail::UnsafeTensorAccessor::getImpl(kernel), stride, padding,
          dilation)));
}

Tensor conv2d(const Tensor& input, const Tensor& kernel, size_t stride,
              size_t padding, size_t dilation) {
  LMP_CHECK(input.device() == kernel.device())
      << "Tensors are on different devices";
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(conv2d_stub()(
          input.device(), *detail::UnsafeTensorAccessor::getImpl(input),
          *detail::UnsafeTensorAccessor::getImpl(kernel), stride, padding,
          dilation)));
}

Tensor conv3d(const Tensor& input, const Tensor& kernel, size_t stride,
              size_t padding, size_t dilation) {
  LMP_CHECK(input.device() == kernel.device())
      << "Tensors are on different devices";
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(conv3d_stub()(
          input.device(), *detail::UnsafeTensorAccessor::getImpl(input),
          *detail::UnsafeTensorAccessor::getImpl(kernel), stride, padding,
          dilation)));
}

}  // namespace lmp::tensor::ops