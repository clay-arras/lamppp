#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using add_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using sub_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using mul_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using div_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using eq_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using ne_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using ge_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using le_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using gt_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using lt_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);

LMP_DECLARE_DISPATCH(add_fn, add_stub);
LMP_DECLARE_DISPATCH(sub_fn, sub_stub);
LMP_DECLARE_DISPATCH(mul_fn, mul_stub);
LMP_DECLARE_DISPATCH(div_fn, div_stub);
LMP_DECLARE_DISPATCH(eq_fn, eq_stub);
LMP_DECLARE_DISPATCH(ne_fn, ne_stub);
LMP_DECLARE_DISPATCH(ge_fn, ge_stub);
LMP_DECLARE_DISPATCH(le_fn, le_stub);
LMP_DECLARE_DISPATCH(gt_fn, gt_stub);
LMP_DECLARE_DISPATCH(lt_fn, lt_stub);

TensorImpl add_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl div_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl mul_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl sub_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl eq_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl ge_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl gt_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl le_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl lt_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl ne_cpu(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(add_stub, DeviceType::CPU, add_cpu);
LMP_REGISTER_DISPATCH(div_stub, DeviceType::CPU, div_cpu);
LMP_REGISTER_DISPATCH(mul_stub, DeviceType::CPU, mul_cpu);
LMP_REGISTER_DISPATCH(sub_stub, DeviceType::CPU, sub_cpu);
LMP_REGISTER_DISPATCH(eq_stub, DeviceType::CPU, eq_cpu);
LMP_REGISTER_DISPATCH(ge_stub, DeviceType::CPU, ge_cpu);
LMP_REGISTER_DISPATCH(gt_stub, DeviceType::CPU, gt_cpu);
LMP_REGISTER_DISPATCH(le_stub, DeviceType::CPU, le_cpu);
LMP_REGISTER_DISPATCH(lt_stub, DeviceType::CPU, lt_cpu);
LMP_REGISTER_DISPATCH(ne_stub, DeviceType::CPU, ne_cpu);

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

Tensor equal(const Tensor& a, const Tensor& b);
Tensor not_equal(const Tensor& a, const Tensor& b);
Tensor greater_equal(const Tensor& a, const Tensor& b);
Tensor less_equal(const Tensor& a, const Tensor& b);
Tensor greater(const Tensor& a, const Tensor& b);
Tensor less(const Tensor& a, const Tensor& b);

}  // namespace lmp::tensor::ops