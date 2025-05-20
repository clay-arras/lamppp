#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using eq_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using ne_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using ge_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using le_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using gt_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using lt_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);

LMP_DECLARE_DISPATCH(eq_fn, eq_stub);
LMP_DECLARE_DISPATCH(ne_fn, ne_stub);
LMP_DECLARE_DISPATCH(ge_fn, ge_stub);
LMP_DECLARE_DISPATCH(le_fn, le_stub);
LMP_DECLARE_DISPATCH(gt_fn, gt_stub);
LMP_DECLARE_DISPATCH(lt_fn, lt_stub);

TensorImpl eq_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl ge_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl gt_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl le_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl lt_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl ne_cpu(const TensorImpl& a, const TensorImpl& b);

LMP_REGISTER_DISPATCH(eq_stub, DeviceType::CPU, eq_cpu);
LMP_REGISTER_DISPATCH(ge_stub, DeviceType::CPU, ge_cpu);
LMP_REGISTER_DISPATCH(gt_stub, DeviceType::CPU, gt_cpu);
LMP_REGISTER_DISPATCH(le_stub, DeviceType::CPU, le_cpu);
LMP_REGISTER_DISPATCH(lt_stub, DeviceType::CPU, lt_cpu);
LMP_REGISTER_DISPATCH(ne_stub, DeviceType::CPU, ne_cpu);

Tensor equal(const Tensor& a, const Tensor& b);
Tensor not_equal(const Tensor& a, const Tensor& b);
Tensor greater_equal(const Tensor& a, const Tensor& b);
Tensor less_equal(const Tensor& a, const Tensor& b);
Tensor greater(const Tensor& a, const Tensor& b);
Tensor less(const Tensor& a, const Tensor& b);

}  // namespace lmp::tensor::ops