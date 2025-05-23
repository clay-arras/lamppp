#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cuda/expand.cuh"
#include "lamppp/tensor/cuda/kernels.cuh"
#include "lamppp/tensor/cuda/reduct.cuh"
#include "lamppp/tensor/cuda/unary.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

TensorImpl add_cuda(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<AddFunctor>(meta);
  return meta.out();
}

TensorImpl sub_cuda(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<SubFunctor>(meta);
  return meta.out();
}

TensorImpl mul_cuda(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<MulFunctor>(meta);
  return meta.out();
}

TensorImpl div_cuda(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<DivFunctor>(meta);
  return meta.out();
}

TensorImpl eq_cuda(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<EqFunctor>(meta);
  return meta.out();
}

TensorImpl ne_cuda(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<NeFunctor>(meta);
  return meta.out();
}

TensorImpl le_cuda(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<LeFunctor>(meta);
  return meta.out();
}

TensorImpl lt_cuda(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<LtFunctor>(meta);
  return meta.out();
}

TensorImpl ge_cuda(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<GeFunctor>(meta);
  return meta.out();
}

TensorImpl gt_cuda(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<GtFunctor>(meta);
  return meta.out();
}

TensorImpl log_cuda(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<LogFunctor>(meta);
  return meta.out();
}

TensorImpl exp_cuda(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<ExpFunctor>(meta);
  return meta.out();
}

TensorImpl sqrt_cuda(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<SqrtFunctor>(meta);
  return meta.out();
}

TensorImpl abs_cuda(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<AbsFunctor>(meta);
  return meta.out();
}

TensorImpl sin_cuda(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<SinFunctor>(meta);
  return meta.out();
}

TensorImpl cos_cuda(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<CosFunctor>(meta);
  return meta.out();
}

TensorImpl tan_cuda(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<TanFunctor>(meta);
  return meta.out();
}

TensorImpl clamp_cuda(const TensorImpl& a, Scalar min_val, Scalar max_val) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<ClampFunctor>(meta, min_val, max_val);
  return meta.out();
}

TensorImpl sum_cuda(const TensorImpl& a, size_t axis) {
  TensorMetaHandler meta(&a, axis);
  reduct_dispatch_handler<SumFunctor>(meta, axis);
  return meta.out();
}

TensorImpl max_cuda(const TensorImpl& a, size_t axis) {
  TensorMetaHandler meta(&a, axis);
  reduct_dispatch_handler<MaxFunctor>(meta, axis);
  return meta.out();
}

TensorImpl min_cuda(const TensorImpl& a, size_t axis) {
  TensorMetaHandler meta(&a, axis);
  reduct_dispatch_handler<MinFunctor>(meta, axis);
  return meta.out();
}

LMP_REGISTER_DISPATCH(ops::add_stub, DeviceType::CUDA, add_cuda);
LMP_REGISTER_DISPATCH(ops::sub_stub, DeviceType::CUDA, sub_cuda);
LMP_REGISTER_DISPATCH(ops::mul_stub, DeviceType::CUDA, mul_cuda);
LMP_REGISTER_DISPATCH(ops::div_stub, DeviceType::CUDA, div_cuda);

LMP_REGISTER_DISPATCH(ops::eq_stub, DeviceType::CUDA, eq_cuda);
LMP_REGISTER_DISPATCH(ops::ne_stub, DeviceType::CUDA, ne_cuda);
LMP_REGISTER_DISPATCH(ops::le_stub, DeviceType::CUDA, le_cuda);
LMP_REGISTER_DISPATCH(ops::lt_stub, DeviceType::CUDA, lt_cuda);
LMP_REGISTER_DISPATCH(ops::ge_stub, DeviceType::CUDA, ge_cuda);
LMP_REGISTER_DISPATCH(ops::gt_stub, DeviceType::CUDA, gt_cuda);

LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CUDA, log_cuda);
LMP_REGISTER_DISPATCH(ops::exp_stub, DeviceType::CUDA, exp_cuda);
LMP_REGISTER_DISPATCH(ops::sqrt_stub, DeviceType::CUDA, sqrt_cuda);
LMP_REGISTER_DISPATCH(ops::abs_stub, DeviceType::CUDA, abs_cuda);
LMP_REGISTER_DISPATCH(ops::sin_stub, DeviceType::CUDA, sin_cuda);
LMP_REGISTER_DISPATCH(ops::cos_stub, DeviceType::CUDA, cos_cuda);
LMP_REGISTER_DISPATCH(ops::tan_stub, DeviceType::CUDA, tan_cuda);
LMP_REGISTER_DISPATCH(ops::clamp_stub, DeviceType::CUDA, clamp_cuda);

LMP_REGISTER_DISPATCH(ops::sum_stub, DeviceType::CUDA, sum_cuda);
LMP_REGISTER_DISPATCH(ops::max_stub, DeviceType::CUDA, max_cuda);
LMP_REGISTER_DISPATCH(ops::min_stub, DeviceType::CUDA, min_cuda);

}  // namespace lmp::tensor::detail::cuda