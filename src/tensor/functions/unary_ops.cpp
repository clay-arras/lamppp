#include "include/lamppp/tensor/functions/unary_ops.hpp"
#include "include/lamppp/tensor/cuda/unary_kern.cuh"
#include "include/lamppp/tensor/tensor.hpp"
#include "include/lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(log_stub);
LMP_DEFINE_DISPATCH(exp_stub);
LMP_DEFINE_DISPATCH(sqrt_stub);
LMP_DEFINE_DISPATCH(abs_stub);
LMP_DEFINE_DISPATCH(sin_stub);
LMP_DEFINE_DISPATCH(cos_stub);
LMP_DEFINE_DISPATCH(tan_stub);
LMP_DEFINE_DISPATCH(clamp_stub);

TensorImpl log_cpu(const TensorImpl& a) {
  assert(false && "Not Implemented");
}

TensorImpl log_cuda(const TensorImpl& a) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c(a.size() * sizeof(scalar_t), DeviceType::CUDA);
    ::lmp::tensor::detail::cuda::vecLog<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c.data()), a.size());
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl exp_cpu(const TensorImpl& a) {
  assert(false && "Not Implemented");
}

TensorImpl exp_cuda(const TensorImpl& a) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c(a.size() * sizeof(scalar_t), DeviceType::CUDA);
    ::lmp::tensor::detail::cuda::vecExp<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c.data()), a.size());
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl sqrt_cpu(const TensorImpl& a) {
  assert(false && "CPU sqrt not implemented");
}

TensorImpl sqrt_cuda(const TensorImpl& a) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c(a.size() * sizeof(scalar_t), a.device());
    ::lmp::tensor::detail::cuda::vecSqrt<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c.data()), a.size());
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl abs_cpu(const TensorImpl& a) {
  assert(false && "CPU abs not implemented");
}

TensorImpl abs_cuda(const TensorImpl& a) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c(a.size() * sizeof(scalar_t), a.device());
    ::lmp::tensor::detail::cuda::vecAbs<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c.data()), a.size());
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl sin_cpu(const TensorImpl& a) {
  assert(false && "CPU sin not implemented");
}

TensorImpl sin_cuda(const TensorImpl& a) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c(a.size() * sizeof(scalar_t), a.device());
    ::lmp::tensor::detail::cuda::vecSin<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c.data()), a.size());
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl cos_cpu(const TensorImpl& a) {
  assert(false && "CPU cos not implemented");
}

TensorImpl cos_cuda(const TensorImpl& a) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c(a.size() * sizeof(scalar_t), a.device());
    ::lmp::tensor::detail::cuda::vecCos<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c.data()), a.size());
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl tan_cpu(const TensorImpl& a) {
  assert(false && "CPU tan not implemented");
}

TensorImpl tan_cuda(const TensorImpl& a) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c(a.size() * sizeof(scalar_t), a.device());
    ::lmp::tensor::detail::cuda::vecTan<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c.data()), a.size());
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl clamp_cpu(const TensorImpl& a, Scalar min_val, Scalar max_val) {
  assert(false && "CPU clamp not implemented");
}

TensorImpl clamp_cuda(const TensorImpl& a, Scalar min_s, Scalar max_s) {
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c(a.size() * sizeof(scalar_t), a.device());
    ::lmp::tensor::detail::cuda::vecClamp<scalar_t>(
        static_cast<const scalar_t*>(a.data()), static_cast<scalar_t>(min_s),
        static_cast<scalar_t>(max_s), static_cast<scalar_t*>(c.data()),
        a.size());
    return TensorImpl(c, a.shape(), a.type());
  });
}

Tensor exp(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      exp_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

Tensor log(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      log_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

Tensor sqrt(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sqrt_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

Tensor abs(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      abs_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

Tensor sin(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sin_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

Tensor cos(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      cos_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

Tensor tan(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      tan_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

Tensor clamp(const Tensor& a, Scalar min_val, Scalar max_val) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      clamp_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a), min_val,
                 max_val)));
}

}  // namespace lmp::tensor::ops