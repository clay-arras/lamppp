#include "lamppp/tensor/cuda/kernels.cuh"
#include "lamppp/tensor/cuda/unary.cuh"
#include "lamppp/tensor/cuda/utils.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

TensorImpl log_cuda(const TensorImpl& a) {
  internal::TensorMetaHandler meta({a});
  meta.handle_unary_op();
  unary_dispatch_handler<ExpFunctor>(meta);
  return meta.out();
}

TensorImpl exp_cuda(const TensorImpl& a) {
  internal::TensorMetaHandler meta({a});
  meta.handle_unary_op();
  unary_dispatch_handler<ExpFunctor>(meta);
  return meta.out();
}

TensorImpl sqrt_cuda(const TensorImpl& a) {
  internal::TensorMetaHandler meta({a});
  meta.handle_unary_op();
  unary_dispatch_handler<SqrtFunctor>(meta);
  return meta.out();
}

TensorImpl abs_cuda(const TensorImpl& a) {
  internal::TensorMetaHandler meta({a});
  meta.handle_unary_op();
  unary_dispatch_handler<AbsFunctor>(meta);
  return meta.out();
}

TensorImpl sin_cuda(const TensorImpl& a) {
  internal::TensorMetaHandler meta({a});
  meta.handle_unary_op();
  unary_dispatch_handler<SinFunctor>(meta);
  return meta.out();
}

TensorImpl cos_cuda(const TensorImpl& a) {
  internal::TensorMetaHandler meta({a});
  meta.handle_unary_op();
  unary_dispatch_handler<CosFunctor>(meta);
  return meta.out();
}

TensorImpl tan_cuda(const TensorImpl& a) {
  internal::TensorMetaHandler meta({a});
  meta.handle_unary_op();
  unary_dispatch_handler<TanFunctor>(meta);
  return meta.out();
}

TensorImpl clamp_cuda(const TensorImpl& a, Scalar min_val, Scalar max_val) {
  internal::TensorMetaHandler meta({a});
  meta.handle_unary_op();
  unary_dispatch_handler<ClampFunctor>(meta, min_val, max_val);
  return meta.out();
}

}  // namespace lmp::tensor::detail::cuda