#include "lamppp/tensor/functions/reduct_ops.hpp"
#include "lamppp/tensor/cuda/reduct_kern.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(sum_stub);
LMP_DEFINE_DISPATCH(max_stub);
LMP_DEFINE_DISPATCH(min_stub);

TensorImpl sum_cpu(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

TensorImpl max_cpu(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

TensorImpl min_cpu(const TensorImpl& a, size_t axis) {
  assert(false && "Not Implemented");
}

TensorImpl sum_cuda(const TensorImpl& a, size_t axis) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    size_t out_size = a.size() / a.shape()[axis];
    Storage c(out_size * sizeof(scalar_t), DeviceType::CUDA);
    ::lmp::tensor::detail::cuda::vecSum<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c.data()), a.shape().data(), a.strides().data(),
        axis, a.shape().size(), a.size());
    std::vector<size_t> nshape = a.shape();
    nshape[axis] = 1;
    return TensorImpl(c, nshape, a.type());
  });
}

TensorImpl max_cuda(const TensorImpl& a, size_t axis) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    size_t out_elem_count = a.size() / a.shape()[axis];
    Storage c(out_elem_count * sizeof(scalar_t), DeviceType::CUDA);
    ::lmp::tensor::detail::cuda::vecMax<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c.data()), a.shape().data(), a.strides().data(),
        axis, a.shape().size(), a.size());
    std::vector<size_t> nshape = a.shape();
    nshape[axis] = 1;
    return TensorImpl(c, nshape, a.type());
  });
}

TensorImpl min_cuda(const TensorImpl& a, size_t axis) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    size_t out_elem_count = a.size() / a.shape()[axis];
    Storage c(out_elem_count * sizeof(scalar_t), DeviceType::CUDA);
    ::lmp::tensor::detail::cuda::vecMin<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c.data()), a.shape().data(), a.strides().data(),
        axis, a.shape().size(), a.size());
    std::vector<size_t> nshape = a.shape();
    nshape[axis] = 1;
    return TensorImpl(c, nshape, a.type());
  });
}

Tensor sum(const Tensor& a, size_t axis) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sum_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), axis)));
}

Tensor max(const Tensor& a, size_t axis) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      max_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), axis)));
}

Tensor min(const Tensor& a, size_t axis) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      min_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), axis)));
}

}  // namespace lmp::tensor::ops
