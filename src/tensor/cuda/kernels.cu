#include "lamppp/tensor/cuda/expand.cuh"
#include "lamppp/tensor/cuda/kernels.cuh"
#include "lamppp/tensor/cuda/meta_util.cuh"
#include "lamppp/tensor/cuda/unary.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

TensorImpl add_cuda(const TensorImpl& a, const TensorImpl& b) {
  internal::TensorMetaHandler meta({a, b});
  meta.handle_expand_op();
  expand_dispatch_handler<AddFunctor>(meta);
  return meta.out();
}

TensorImpl sub_cuda(const TensorImpl& a, const TensorImpl& b) {
  internal::TensorMetaHandler meta({a, b});
  meta.handle_expand_op();
  expand_dispatch_handler<SubFunctor>(meta);
  return meta.out();
}

TensorImpl mul_cuda(const TensorImpl& a, const TensorImpl& b) {
  internal::TensorMetaHandler meta({a, b});
  meta.handle_expand_op();
  expand_dispatch_handler<MulFunctor>(meta);
  return meta.out();
}

TensorImpl div_cuda(const TensorImpl& a, const TensorImpl& b) {
  internal::TensorMetaHandler meta({a, b});
  meta.handle_expand_op();
  expand_dispatch_handler<DivFunctor>(meta);
  return meta.out();
}

TensorImpl eq_cuda(const TensorImpl& a, const TensorImpl& b) {
  internal::TensorMetaHandler meta({a, b});
  meta.handle_expand_op();
  expand_dispatch_handler<EqFunctor>(meta);
  return meta.out();
}

TensorImpl ne_cuda(const TensorImpl& a, const TensorImpl& b) {
  internal::TensorMetaHandler meta({a, b});
  meta.handle_expand_op();
  expand_dispatch_handler<NeFunctor>(meta);
  return meta.out();
}

TensorImpl le_cuda(const TensorImpl& a, const TensorImpl& b) {
  internal::TensorMetaHandler meta({a, b});
  meta.handle_expand_op();
  expand_dispatch_handler<LeFunctor>(meta);
  return meta.out();
}

TensorImpl lt_cuda(const TensorImpl& a, const TensorImpl& b) {
  internal::TensorMetaHandler meta({a, b});
  meta.handle_expand_op();
  expand_dispatch_handler<LtFunctor>(meta);
  return meta.out();
}

TensorImpl ge_cuda(const TensorImpl& a, const TensorImpl& b) {
  internal::TensorMetaHandler meta({a, b});
  meta.handle_expand_op();
  expand_dispatch_handler<GeFunctor>(meta);
  return meta.out();
}

TensorImpl gt_cuda(const TensorImpl& a, const TensorImpl& b) {
  internal::TensorMetaHandler meta({a, b});
  meta.handle_expand_op();
  expand_dispatch_handler<GtFunctor>(meta);
  return meta.out();
}

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