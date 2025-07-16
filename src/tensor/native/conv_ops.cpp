#include "lamppp/tensor/native/conv_ops.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(conv_fn, conv_stub);

Tensor conv(const Tensor& input, const Tensor& kernel, size_t stride,
            size_t padding, size_t dilation) {
  LMP_CHECK(input.device() == kernel.device())
      << "Tensors are on different devices";
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      conv_stub()(input.device(), *detail::UnsafeTensorAccessor::getImpl(input),
                  *detail::UnsafeTensorAccessor::getImpl(kernel), stride,
                  padding, dilation)));
}

}  // namespace lmp::tensor::ops