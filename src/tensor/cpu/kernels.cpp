#include "lamppp/tensor/cpu/kernels.hpp"
#include "lamppp/tensor/cpu/matrix.hpp"
#include "lamppp/tensor/cpu/expand.hpp"
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cpu/reduct.hpp"
#include "lamppp/tensor/cpu/unary.hpp"
#include "lamppp/tensor/native/expand_ops.hpp"
#include "lamppp/tensor/native/reduct_ops.hpp"

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

TensorImpl pow_cpu(const TensorImpl& a, const TensorImpl& b) {
  TensorMetaHandler meta(&a, &b);
  expand_dispatch_handler<PowFunctor>(meta);
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

TensorImpl neg_cpu(const TensorImpl& a) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<NegFunctor>(meta);
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

TensorImpl transpose_cpu(const TensorImpl& a) {
  LMP_CHECK(a.shape().size() == 2) <<
      "Invalid argument, transpose can only be performed on matrices of dim 2";
  size_t m = a.shape()[0];
  size_t n = a.shape()[1];

  DataType out_dtype = a.type();

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c_storage(m * n * sizeof(scalar_t), DeviceType::CPU);
    ::lmp::tensor::detail::cpu::cpuTranspose<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c_storage.data()), m, n);
    return TensorImpl(c_storage, {n, m}, out_dtype);
  });
}

TensorImpl matmul_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_CHECK(a.shape().size() == 2 && b.shape().size() == 2) <<
            "Both matrices must be 2D.";
  LMP_CHECK(a.shape()[1] == b.shape()[0]) <<
            "Incompatible matrix dimensions for multiplication.";

  size_t m = a.shape()[0];
  size_t n = b.shape()[1];
  size_t k = a.shape()[1];

  DataType out_dtype = type_upcast(a.type(), b.type());

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type_t = scalar_t;
        Storage c_storage(m * n * sizeof(out_type_t), DeviceType::CPU);
        ::lmp::tensor::detail::cpu::cpuMatMul<a_type_t, b_type_t, out_type_t>(
            static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type_t*>(c_storage.data()), m, n, k);
        return TensorImpl(c_storage, {m, n}, out_dtype);
      });
    });
  });
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

TensorImpl prod_cpu(const TensorImpl& a, size_t axis) {
  TensorMetaHandler meta(&a, axis);
  reduct_dispatch_handler<ProdFunctor>(meta, axis);
  return meta.out();
}

LMP_REGISTER_DISPATCH(ops::add_stub, DeviceType::CPU, add_cpu);
LMP_REGISTER_DISPATCH(ops::sub_stub, DeviceType::CPU, sub_cpu);
LMP_REGISTER_DISPATCH(ops::mul_stub, DeviceType::CPU, mul_cpu);
LMP_REGISTER_DISPATCH(ops::div_stub, DeviceType::CPU, div_cpu);
LMP_REGISTER_DISPATCH(ops::pow_stub, DeviceType::CPU, pow_cpu);
LMP_REGISTER_DISPATCH(ops::eq_stub, DeviceType::CPU, eq_cpu);
LMP_REGISTER_DISPATCH(ops::ne_stub, DeviceType::CPU, ne_cpu);
LMP_REGISTER_DISPATCH(ops::le_stub, DeviceType::CPU, le_cpu);
LMP_REGISTER_DISPATCH(ops::lt_stub, DeviceType::CPU, lt_cpu);
LMP_REGISTER_DISPATCH(ops::ge_stub, DeviceType::CPU, ge_cpu);
LMP_REGISTER_DISPATCH(ops::gt_stub, DeviceType::CPU, gt_cpu);

LMP_REGISTER_DISPATCH(ops::neg_stub, DeviceType::CPU, neg_cpu);
LMP_REGISTER_DISPATCH(ops::abs_stub, DeviceType::CPU, abs_cpu);
LMP_REGISTER_DISPATCH(ops::clamp_stub, DeviceType::CPU, clamp_cpu);
LMP_REGISTER_DISPATCH(ops::cos_stub, DeviceType::CPU, cos_cpu);
LMP_REGISTER_DISPATCH(ops::exp_stub, DeviceType::CPU, exp_cpu);
LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CPU, log_cpu);
LMP_REGISTER_DISPATCH(ops::sin_stub, DeviceType::CPU, sin_cpu);
LMP_REGISTER_DISPATCH(ops::sqrt_stub, DeviceType::CPU, sqrt_cpu);
LMP_REGISTER_DISPATCH(ops::tan_stub, DeviceType::CPU, tan_cpu);

LMP_REGISTER_DISPATCH(ops::transpose_stub, DeviceType::CPU, transpose_cpu);
LMP_REGISTER_DISPATCH(ops::matmul_stub, DeviceType::CPU, matmul_cpu);

LMP_REGISTER_DISPATCH(ops::sum_stub, DeviceType::CPU, sum_cpu);
LMP_REGISTER_DISPATCH(ops::max_stub, DeviceType::CPU, max_cpu);
LMP_REGISTER_DISPATCH(ops::min_stub, DeviceType::CPU, min_cpu);
LMP_REGISTER_DISPATCH(ops::prod_stub, DeviceType::CPU, prod_cpu);

}  // namespace lmp::tensor::detail::cpu