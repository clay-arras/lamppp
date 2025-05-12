#include "include/lamppp/tensor/functions/unary_ops.hpp"
#include "include/lamppp/tensor/cuda/unary_kern.cuh"
#include "include/lamppp/tensor/tensor.hpp"
#include "include/lamppp/tensor/tensor_impl.hpp"

namespace autograd {

DEFINE_DISPATCH(log_stub);
DEFINE_DISPATCH(exp_stub);
DEFINE_DISPATCH(relu_stub);

TensorImpl log_cpu(const TensorImpl& a) {
  assert(false && "Not Implemented");
}

TensorImpl log_cuda(const TensorImpl& a) {
  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using scalar_t_ = scalar_t;
    Storage c(a.size() * sizeof(scalar_t_), DeviceType::CUDA);
    vecLog<scalar_t_>(a.size(), static_cast<const scalar_t_*>(a.data()),
                      static_cast<scalar_t_*>(c.data()));
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl exp_cpu(const TensorImpl& a) {
  assert(false && "Not Implemented");
}

TensorImpl exp_cuda(const TensorImpl& a) {
  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using scalar_t_ = scalar_t;
    Storage c(a.size() * sizeof(scalar_t_), DeviceType::CUDA);
    vecExp<scalar_t_>(a.size(), static_cast<const scalar_t_*>(a.data()),
                      static_cast<scalar_t_*>(c.data()));
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl relu_cpu(const TensorImpl& a) {
  assert(false && "Not Implemented");
}

TensorImpl relu_cuda(const TensorImpl& a) {
  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using scalar_t_ = scalar_t;
    Storage c(a.size() * sizeof(scalar_t_), DeviceType::CUDA);
    vecRelu<scalar_t_>(a.size(), static_cast<const scalar_t_*>(a.data()),
                       static_cast<scalar_t_*>(c.data()));
    return TensorImpl(c, a.shape(), a.type());
  });
}

Tensor relu(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      relu_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

Tensor exp(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      exp_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

Tensor log(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      log_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

}  // namespace autograd