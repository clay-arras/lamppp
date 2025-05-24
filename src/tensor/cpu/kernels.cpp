#include "lamppp/tensor/cpu/kernels.hpp"
#include "lamppp/tensor/cpu/expand.hpp"
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cpu/reduct.hpp"
#include "lamppp/tensor/cpu/unary.hpp"
#include "lamppp/tensor/functions/expand_ops.hpp"
#include "lamppp/tensor/functions/reduct_ops.hpp"

namespace lmp::tensor::detail::cpu {

TensorImpl add_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<AddFunctor>(meta);
  return meta.out();
}

TensorImpl sub_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<SubFunctor>(meta);
  return meta.out();
}

TensorImpl mul_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<MulFunctor>(meta);
  return meta.out();
}

TensorImpl div_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<DivFunctor>(meta);
  return meta.out();
}

TensorImpl eq_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<EqFunctor>(meta);
  return meta.out();
}

TensorImpl ne_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<NeFunctor>(meta);
  return meta.out();
}

TensorImpl le_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<LeFunctor>(meta);
  return meta.out();
}

TensorImpl lt_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<LtFunctor>(meta);
  return meta.out();
}

TensorImpl ge_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<GeFunctor>(meta);
  return meta.out();
}

TensorImpl gt_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<GtFunctor>(meta);
  return meta.out();
}

TensorImpl log_cpu(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<LogFunctor>(meta);
  return meta.out();
}

TensorImpl exp_cpu(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<ExpFunctor>(meta);
  return meta.out();
}

TensorImpl sqrt_cpu(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<SqrtFunctor>(meta);
  return meta.out();
}

TensorImpl abs_cpu(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<AbsFunctor>(meta);
  return meta.out();
}

TensorImpl sin_cpu(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<SinFunctor>(meta);
  return meta.out();
}

TensorImpl cos_cpu(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<CosFunctor>(meta);
  return meta.out();
}

TensorImpl tan_cpu(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<TanFunctor>(meta);
  return meta.out();
}

TensorImpl clamp_cpu(const TensorImpl& a, Scalar min_val, Scalar max_val) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<ClampFunctor>(meta, min_val, max_val);
  return meta.out();
}

TensorImpl sum_cpu(const TensorImpl& a, size_t axis) {
  TensorMetaHandler meta(&a, axis);
  reduct_dispatch_handler<SumFunctor>(meta, axis);
  return meta.out();
}

TensorImpl max_cpu(const TensorImpl& a, size_t axis) {
  TensorMetaHandler meta(&a, axis);
  reduct_dispatch_handler<MaxFunctor>(meta, axis);
  return meta.out();
}

TensorImpl min_cpu(const TensorImpl& a, size_t axis) {
  TensorMetaHandler meta(&a, axis);
  reduct_dispatch_handler<MinFunctor>(meta, axis);
  return meta.out();
}

LMP_REGISTER_DISPATCH(ops::add_stub, DeviceType::CPU, add_cpu);
LMP_REGISTER_DISPATCH(ops::sub_stub, DeviceType::CPU, sub_cpu);
LMP_REGISTER_DISPATCH(ops::mul_stub, DeviceType::CPU, mul_cpu);
LMP_REGISTER_DISPATCH(ops::div_stub, DeviceType::CPU, div_cpu);
LMP_REGISTER_DISPATCH(ops::eq_stub, DeviceType::CPU, eq_cpu);
LMP_REGISTER_DISPATCH(ops::ne_stub, DeviceType::CPU, ne_cpu);
LMP_REGISTER_DISPATCH(ops::le_stub, DeviceType::CPU, le_cpu);
LMP_REGISTER_DISPATCH(ops::lt_stub, DeviceType::CPU, lt_cpu);
LMP_REGISTER_DISPATCH(ops::ge_stub, DeviceType::CPU, ge_cpu);
LMP_REGISTER_DISPATCH(ops::gt_stub, DeviceType::CPU, gt_cpu);

LMP_REGISTER_DISPATCH(ops::abs_stub, DeviceType::CPU, abs_cpu);
LMP_REGISTER_DISPATCH(ops::clamp_stub, DeviceType::CPU, clamp_cpu);
LMP_REGISTER_DISPATCH(ops::cos_stub, DeviceType::CPU, cos_cpu);
LMP_REGISTER_DISPATCH(ops::exp_stub, DeviceType::CPU, exp_cpu);
LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CPU, log_cpu);
LMP_REGISTER_DISPATCH(ops::sin_stub, DeviceType::CPU, sin_cpu);
LMP_REGISTER_DISPATCH(ops::sqrt_stub, DeviceType::CPU, sqrt_cpu);
LMP_REGISTER_DISPATCH(ops::tan_stub, DeviceType::CPU, tan_cpu);

LMP_REGISTER_DISPATCH(ops::sum_stub, DeviceType::CPU, sum_cpu);
LMP_REGISTER_DISPATCH(ops::max_stub, DeviceType::CPU, max_cpu);
LMP_REGISTER_DISPATCH(ops::min_stub, DeviceType::CPU, min_cpu);

}  // namespace lmp::tensor::detail::cpu