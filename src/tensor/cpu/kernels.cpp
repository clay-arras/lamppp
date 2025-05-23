#include "lamppp/tensor/cpu/kernels.hpp"
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cpu/unary.hpp"

namespace lmp::tensor::detail::cpu {

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

LMP_REGISTER_DISPATCH(ops::abs_stub, DeviceType::CPU, abs_cpu);
LMP_REGISTER_DISPATCH(ops::clamp_stub, DeviceType::CPU, clamp_cpu);
LMP_REGISTER_DISPATCH(ops::cos_stub, DeviceType::CPU, cos_cpu);
LMP_REGISTER_DISPATCH(ops::exp_stub, DeviceType::CPU, exp_cpu);
LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CPU, log_cpu);
LMP_REGISTER_DISPATCH(ops::sin_stub, DeviceType::CPU, sin_cpu);
LMP_REGISTER_DISPATCH(ops::sqrt_stub, DeviceType::CPU, sqrt_cpu);
LMP_REGISTER_DISPATCH(ops::tan_stub, DeviceType::CPU, tan_cpu);

}  // namespace lmp::tensor::detail::cpu